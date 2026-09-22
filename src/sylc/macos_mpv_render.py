"""Render libmpv video directly into SyLC's Qt OpenGL surface on macOS."""

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtOpenGLWidgets import QOpenGLWidget

logger = logging.getLogger(__name__)


class MacOSMpvVideoWidget(QOpenGLWidget):
    frame_ready = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._render_context = None
        self.setMouseTracking(True)
        self.frame_ready.connect(self.update, Qt.ConnectionType.QueuedConnection)

    def attach_player(self, player, mpv_module):
        """Create the render context while Qt's OpenGL context is current."""
        self.detach_player()
        self.makeCurrent()
        try:
            context = self.context()
            if context is None:
                raise RuntimeError("Qt OpenGL context is unavailable")

            def get_proc_address(_ctx, name):
                return int(context.getProcAddress(name.decode('ascii')) or 0)

            self._get_proc_address = mpv_module.MpvGlGetProcAddressFn(get_proc_address)
            self._render_context = mpv_module.MpvRenderContext(
                player, 'opengl',
                opengl_init_params={'get_proc_address': self._get_proc_address},
            )
            # libmpv calls this on its render thread; the signal queues update()
            # onto Qt's GUI thread where paintGL owns the OpenGL context.
            self._render_context.update_cb = self.frame_ready.emit
        finally:
            self.doneCurrent()
        self.update()

    def detach_player(self):
        render_context = self._render_context
        if render_context is None:
            return
        self._render_context = None
        render_context.update_cb = None
        self.makeCurrent()
        try:
            render_context.free()
        finally:
            self.doneCurrent()
            self._get_proc_address = None

    def paintGL(self):
        render_context = self._render_context
        if render_context is None:
            return
        scale = self.devicePixelRatioF()
        try:
            render_context.update()
            render_context.render(
                opengl_fbo={
                    'fbo': self.defaultFramebufferObject(),
                    'w': max(1, round(self.width() * scale)),
                    'h': max(1, round(self.height() * scale)),
                },
                flip_y=True,
            )
            render_context.report_swap()
        except Exception:
            logger.exception("[MPV] Failed to render video into the macOS widget")
