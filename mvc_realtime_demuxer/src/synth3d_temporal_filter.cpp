#include "synth3d_temporal_filter.h"

#include "parallel_chunks.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

float clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

float smoothstep(float lo, float hi, float value) {
    const float t = clamp01((value - lo) / std::max(1.0e-8f, hi - lo));
    return t * t * (3.0f - 2.0f * t);
}

float time_scale(float source_dt_ms, float reference_dt_ms) {
    if (!(source_dt_ms > 0.0f) || !std::isfinite(source_dt_ms))
        source_dt_ms = reference_dt_ms;
    return std::max(0.10f, std::min(12.0f,
        source_dt_ms / reference_dt_ms));
}

float time_true_alpha(float alpha, float scale) {
    alpha = clamp01(alpha);
    if (std::abs(scale - 1.0f) < 1.0e-5f) return alpha;
    return 1.0f - std::pow(1.0f - alpha, scale);
}

template <typename Sample>
float bilinear_sample(int width, int height, float x, float y,
                      const Sample& sample) {
    x = std::max(0.0f, std::min(static_cast<float>(width - 1), x));
    y = std::max(0.0f, std::min(static_cast<float>(height - 1), y));
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int x1 = std::min(width - 1, x0 + 1);
    const int y1 = std::min(height - 1, y0 + 1);
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const float a = sample(y0 * width + x0) * (1.0f - fx) +
                    sample(y0 * width + x1) * fx;
    const float b = sample(y1 * width + x0) * (1.0f - fx) +
                    sample(y1 * width + x1) * fx;
    return a * (1.0f - fy) + b * fy;
}

}  // namespace

DepthMultiscaleFilter::DepthMultiscaleFilter(int width, int height)
    : width_(width), height_(height),
      n_(static_cast<size_t>(width) * static_cast<size_t>(height)) {
    if (width <= 0 || height <= 0)
        throw std::invalid_argument("DepthMultiscaleFilter: invalid grid");
    make_kernel(1.15f, kernel_fine_);
    make_kernel(3.20f, kernel_coarse_);
    current_.resize(n_);
    aligned_.resize(n_);
    blur_tmp_.resize(n_);
    low1_.resize(n_);
    low2_.resize(n_);
    high_state_.resize(n_);
    mid_state_.resize(n_);
    low_state_.resize(n_);
    current_high_.resize(n_);
    current_mid_.resize(n_);
    current_low_.resize(n_);
    previous_high_.resize(n_);
    previous_mid_.resize(n_);
    previous_low_.resize(n_);
}

void DepthMultiscaleFilter::reset() {
    primed_ = false;
}

void DepthMultiscaleFilter::make_kernel(
        float sigma, std::vector<float>& kernel) {
    const int radius = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
    kernel.resize(static_cast<size_t>(2 * radius + 1));
    float sum = 0.0f;
    for (int x = -radius; x <= radius; ++x) {
        const float value = std::exp(
            -0.5f * static_cast<float>(x * x) / (sigma * sigma));
        kernel[static_cast<size_t>(x + radius)] = value;
        sum += value;
    }
    for (float& value : kernel) value /= sum;
}

void DepthMultiscaleFilter::gaussian_blur(
        const std::vector<float>& input, std::vector<float>& output,
        const std::vector<float>& kernel) {
    const int radius = static_cast<int>(kernel.size() / 2);
    parallel_chunks(height_, worker_threads, [&](int, int y_begin, int y_end) {
        for (int y = y_begin; y < y_end; ++y) {
            for (int x = 0; x < width_; ++x) {
                float sum = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    const int sx = std::max(0, std::min(width_ - 1, x + k));
                    sum += kernel[static_cast<size_t>(k + radius)] *
                           input[static_cast<size_t>(y) * width_ + sx];
                }
                blur_tmp_[static_cast<size_t>(y) * width_ + x] = sum;
            }
        }
    });
    parallel_chunks(height_, worker_threads, [&](int, int y_begin, int y_end) {
        for (int y = y_begin; y < y_end; ++y) {
            for (int x = 0; x < width_; ++x) {
                float sum = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    const int sy = std::max(0, std::min(height_ - 1, y + k));
                    sum += kernel[static_cast<size_t>(k + radius)] *
                           blur_tmp_[static_cast<size_t>(sy) * width_ + x];
                }
                output[static_cast<size_t>(y) * width_ + x] = sum;
            }
        }
    });
}

void DepthMultiscaleFilter::split_bands(
        const std::vector<float>& input,
        std::vector<float>& high, std::vector<float>& mid,
        std::vector<float>& low) {
    gaussian_blur(input, low1_, kernel_fine_);
    gaussian_blur(low1_, low2_, kernel_coarse_);
    parallel_chunks(static_cast<int>(n_), worker_threads,
                    [&](int, int begin, int end) {
        for (int i = begin; i < end; ++i) {
            const size_t p = static_cast<size_t>(i);
            high[p] = input[p] - low1_[p];
            mid[p] = low1_[p] - low2_[p];
            low[p] = low2_[p];
        }
    });
}

void DepthMultiscaleFilter::transport(
        const std::vector<float>& input, std::vector<float>& output,
        const float* flow_x, const float* flow_y) {
    parallel_chunks(height_, worker_threads, [&](int, int y_begin, int y_end) {
        for (int y = y_begin; y < y_end; ++y) {
            for (int x = 0; x < width_; ++x) {
                const size_t p = static_cast<size_t>(y) * width_ + x;
                const float sx = static_cast<float>(x) -
                    (flow_x ? flow_x[p] : 0.0f);
                const float sy = static_cast<float>(y) -
                    (flow_y ? flow_y[p] : 0.0f);
                output[p] = bilinear_sample(
                    width_, height_, sx, sy,
                    [&](int index) { return input[static_cast<size_t>(index)]; });
            }
        }
    });
}

void DepthMultiscaleFilter::apply(
        uint16_t* depth_q16,
        const float* flow_x, const float* flow_y,
        const float* reliability,
        const float* confidence,
        const float* boundary,
        float source_dt_ms) {
    if (!depth_q16) return;
    parallel_chunks(static_cast<int>(n_), worker_threads,
                    [&](int, int begin, int end) {
        for (int i = begin; i < end; ++i)
            current_[static_cast<size_t>(i)] =
                depth_q16[i] * (1.0f / 65535.0f);
    });

    if (!primed_) {
        split_bands(current_, high_state_, mid_state_, low_state_);
        primed_ = true;
        return;
    }

    transport(high_state_, previous_high_, flow_x, flow_y);
    transport(mid_state_, previous_mid_, flow_x, flow_y);
    transport(low_state_, previous_low_, flow_x, flow_y);

    // Remove the global affine gauge on reliable, confident interiors before
    // deciding whether any individual spatial band truly changed.
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xx = 0.0;
    double sum_xy = 0.0;
    size_t fit_count = 0;
    for (size_t i = 0; i < n_; ++i) {
        const float rel = reliability ? clamp01(reliability[i]) : 1.0f;
        const float conf = confidence ? clamp01(confidence[i]) : 1.0f;
        const float edge = boundary ? clamp01(boundary[i]) : 0.0f;
        if (rel < 0.65f || conf < 0.30f || edge >= 0.25f) continue;
        const double x = current_[i];
        const double y = previous_high_[i] + previous_mid_[i] +
                         previous_low_[i];
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
        ++fit_count;
    }
    float scale = 1.0f;
    float shift = 0.0f;
    if (fit_count >= 256) {
        const double inv = 1.0 / static_cast<double>(fit_count);
        const double mean_x = sum_x * inv;
        const double mean_y = sum_y * inv;
        const double variance = sum_xx * inv - mean_x * mean_x;
        const double covariance = sum_xy * inv - mean_x * mean_y;
        scale = static_cast<float>(covariance / std::max(1.0e-8, variance));
        scale = std::max(0.82f, std::min(1.22f, scale));
        shift = static_cast<float>(mean_y - scale * mean_x);
        shift = std::max(-0.10f, std::min(0.10f, shift));
    }
    parallel_chunks(static_cast<int>(n_), worker_threads,
                    [&](int, int begin, int end) {
        for (int i = begin; i < end; ++i) {
            const size_t p = static_cast<size_t>(i);
            aligned_[p] = clamp01(scale * current_[p] + shift);
        }
    });
    split_bands(aligned_, current_high_, current_mid_, current_low_);

    const float ts = time_scale(source_dt_ms, kReferenceDtMs);
    parallel_chunks(static_cast<int>(n_), worker_threads,
                    [&](int, int begin, int end) {
        for (int i = begin; i < end; ++i) {
            const size_t p = static_cast<size_t>(i);
            const float rel = reliability ? clamp01(reliability[p]) : 1.0f;
            const float uncertain = smoothstep(0.28f, 0.62f, 1.0f - rel);
            auto update = [&](float current, float previous,
                              float base_alpha, float change_lo,
                              float change_hi) {
                const float change = smoothstep(
                    change_lo, change_hi, std::abs(current - previous));
                float alpha = base_alpha + (1.0f - base_alpha) *
                    std::max(uncertain, change);
                alpha = time_true_alpha(alpha, ts);
                return previous + alpha * (current - previous);
            };
            low_state_[p] = update(
                current_low_[p], previous_low_[p], 0.045f, 0.014f, 0.050f);
            mid_state_[p] = update(
                current_mid_[p], previous_mid_[p], 0.16f, 0.010f, 0.038f);
            high_state_[p] = update(
                current_high_[p], previous_high_[p], 0.58f, 0.008f, 0.030f);
            const float reconstructed = clamp01(
                low_state_[p] + mid_state_[p] + high_state_[p]);
            depth_q16[p] = static_cast<uint16_t>(
                reconstructed * 65535.0f + 0.5f);
        }
    });
}

SafetyTemporalFilter::SafetyTemporalFilter(int width, int height)
    : width_(width), height_(height),
      n_(static_cast<size_t>(width) * static_cast<size_t>(height)),
      history_(n_, 0.0f), transported_(n_, 0.0f) {
    if (width <= 0 || height <= 0)
        throw std::invalid_argument("SafetyTemporalFilter: invalid grid");
}

void SafetyTemporalFilter::reset() {
    primed_ = false;
}

void SafetyTemporalFilter::apply(
        uint16_t* safety, size_t stride,
        const float* flow_x, const float* flow_y,
        float source_dt_ms) {
    if (!safety || stride == 0) return;
    if (!primed_) {
        parallel_chunks(static_cast<int>(n_), worker_threads,
                        [&](int, int begin, int end) {
            for (int i = begin; i < end; ++i)
                history_[static_cast<size_t>(i)] =
                    safety[static_cast<size_t>(i) * stride] / 65535.0f;
        });
        primed_ = true;
        return;
    }

    parallel_chunks(height_, worker_threads, [&](int, int y_begin, int y_end) {
        for (int y = y_begin; y < y_end; ++y) {
            for (int x = 0; x < width_; ++x) {
                const size_t p = static_cast<size_t>(y) * width_ + x;
                const float sx = static_cast<float>(x) -
                    (flow_x ? flow_x[p] : 0.0f);
                const float sy = static_cast<float>(y) -
                    (flow_y ? flow_y[p] : 0.0f);
                transported_[p] = bilinear_sample(
                    width_, height_, sx, sy,
                    [&](int index) { return history_[static_cast<size_t>(index)]; });
            }
        }
    });
    const float rise = std::max(0.0f, rise_per_reference) *
        time_scale(source_dt_ms, kReferenceDtMs);
    parallel_chunks(static_cast<int>(n_), worker_threads,
                    [&](int, int begin, int end) {
        for (int i = begin; i < end; ++i) {
            const size_t p = static_cast<size_t>(i);
            const float current = safety[p * stride] / 65535.0f;
            const float filtered = std::min(current, transported_[p] + rise);
            history_[p] = filtered;
            safety[p * stride] = static_cast<uint16_t>(
                clamp01(filtered) * 65535.0f + 0.5f);
        }
    });
}
