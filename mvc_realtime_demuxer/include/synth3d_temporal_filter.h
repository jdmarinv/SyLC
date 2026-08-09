// Causal post-normalization filters validated on the native Synth3D grid.
//
// DepthMultiscaleFilter separates normalized nearness into three Laplacian
// bands.  Each band gets its own temporal response after optical-flow
// transport: broad geometry is deliberately slow, fine detail stays quick.
// A reliable-interior affine fit removes the monocular model's arbitrary
// scale/offset gauge before local temporal changes are evaluated.
//
// SafetyTemporalFilter is the CPU fallback for the renderer's equivalent GPU
// pass.  It is conservative by construction: unsafe changes attack at once;
// recovery is bounded, and the result can never exceed the current ownership
// analysis.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

class DepthMultiscaleFilter {
public:
    static constexpr float kReferenceDtMs = 1000.0f * 1001.0f / 24000.0f;

    DepthMultiscaleFilter(int width, int height);
    void reset();
    int width() const { return width_; }
    int height() const { return height_; }

    // Filters depth_q16 in place. flow is previous->current displacement in
    // grid pixels at each current destination. Null flow means identity.
    void apply(uint16_t* depth_q16,
               const float* flow_x, const float* flow_y,
               const float* reliability,
               const float* confidence,
               const float* boundary,
               float source_dt_ms);

    int worker_threads = 1;

private:
    void make_kernel(float sigma, std::vector<float>& kernel);
    void gaussian_blur(const std::vector<float>& input,
                       std::vector<float>& output,
                       const std::vector<float>& kernel);
    void split_bands(const std::vector<float>& input,
                     std::vector<float>& high,
                     std::vector<float>& mid,
                     std::vector<float>& low);
    void transport(const std::vector<float>& input,
                   std::vector<float>& output,
                   const float* flow_x, const float* flow_y);

    int width_ = 0;
    int height_ = 0;
    size_t n_ = 0;
    bool primed_ = false;
    std::vector<float> kernel_fine_;
    std::vector<float> kernel_coarse_;
    std::vector<float> current_;
    std::vector<float> aligned_;
    std::vector<float> blur_tmp_;
    std::vector<float> low1_;
    std::vector<float> low2_;
    std::vector<float> high_state_;
    std::vector<float> mid_state_;
    std::vector<float> low_state_;
    std::vector<float> current_high_;
    std::vector<float> current_mid_;
    std::vector<float> current_low_;
    std::vector<float> previous_high_;
    std::vector<float> previous_mid_;
    std::vector<float> previous_low_;
};

class SafetyTemporalFilter {
public:
    static constexpr float kReferenceDtMs =
        DepthMultiscaleFilter::kReferenceDtMs;

    SafetyTemporalFilter(int width, int height);
    void reset();
    int width() const { return width_; }
    int height() const { return height_; }

    // `safety` may be interleaved; stride is in uint16 elements. The result is
    // written in place and is always <= the current input value.
    void apply(uint16_t* safety, size_t stride,
               const float* flow_x, const float* flow_y,
               float source_dt_ms);

    float rise_per_reference = 0.012f;
    int worker_threads = 1;

private:
    int width_ = 0;
    int height_ = 0;
    size_t n_ = 0;
    bool primed_ = false;
    std::vector<float> history_;
    std::vector<float> transported_;
};
