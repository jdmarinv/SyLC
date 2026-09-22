"""macOS helper for embedding mpv's standalone window into PySide6 QWidget.

On macOS, mpv's macvk (Vulkan via MoltenVK) backend does not implement native
embedding into --wid (mpv issue #13608). Instead, macvk creates its own Cocoa
NSWindow.

This module provides Cocoa runtime helpers to:
1. Locate mpv's NSWindow.
2. Reparent its CAMetalLayer contentView into the QWidget's native NSView.
3. Order out the empty mpv NSWindow so it no longer floats as a detached box.
"""

import sys
import logging

logger = logging.getLogger(__name__)


def reparent_macos_mpv_view(video_widget) -> bool:
    """Reparent mpv's Cocoa contentView into video_widget's NSView.
    
    Returns True if successfully reparented, False otherwise.
    Safe no-op on non-macOS platforms.
    """
    if sys.platform != 'darwin' or video_widget is None:
        return False

    try:
        import ctypes
        import ctypes.util

        objc_lib = ctypes.util.find_library('objc')
        if not objc_lib:
            return False

        objc = ctypes.cdll.LoadLibrary(objc_lib)
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.objc_msgSend.restype = ctypes.c_void_p
        objc.object_getClassName.restype = ctypes.c_char_p
        objc.object_getClassName.argtypes = [ctypes.c_void_p]

        parent_winid = int(video_widget.winId())
        if not parent_winid:
            return False
        parent_view = ctypes.c_void_p(parent_winid)

        NSApp = objc.objc_msgSend(
            objc.objc_getClass(b"NSApplication"),
            objc.sel_registerName(b"sharedApplication")
        )
        if not NSApp:
            return False

        windows = objc.objc_msgSend(NSApp, objc.sel_registerName(b"windows"))
        if not windows:
            return False

        count = objc.objc_msgSend(windows, objc.sel_registerName(b"count"))
        parent_window = objc.objc_msgSend(parent_view, objc.sel_registerName(b"window"))

        for i in range(count):
            nswin = objc.objc_msgSend(
                windows,
                objc.sel_registerName(b"objectAtIndex:"),
                ctypes.c_ulong(i)
            )
            if not nswin or nswin == parent_window:
                continue

            cname = objc.object_getClassName(nswin).decode('utf-8', 'ignore')
            # SyLC overlay tool windows use QNSWindow. mpv creates a window with class Window or NSWindow
            if cname in ('Window', 'NSWindow') or 'mpv' in cname.lower():
                delegate = objc.objc_msgSend(nswin, objc.sel_registerName(b"delegate"))
                del_cname = objc.object_getClassName(delegate).decode('utf-8', 'ignore') if delegate else ''
                if 'QNS' in del_cname:
                    continue

                content_view = objc.objc_msgSend(nswin, objc.sel_registerName(b"contentView"))
                if not content_view:
                    continue

                # Add content_view as subview of video_widget's NSView
                objc.objc_msgSend(
                    parent_view,
                    objc.sel_registerName(b"addSubview:"),
                    content_view
                )

                # NSViewWidthSizable(2) | NSViewHeightSizable(16) = 18
                objc.objc_msgSend(
                    content_view,
                    objc.sel_registerName(b"setAutoresizingMask:"),
                    ctypes.c_ulong(18)
                )

                # Hide the empty mpv window
                objc.objc_msgSend(nswin, objc.sel_registerName(b"orderOut:"), None)
                logger.info("[MACOS-EMBED] Successfully reparented MPV view into video widget")
                return True

    except Exception as e:
        logger.warning(f"[MACOS-EMBED] Error during Cocoa reparenting: {e}")

    return False
