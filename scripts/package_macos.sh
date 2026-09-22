#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
PACKAGER="${SYLC_PYINSTALLER:-$ROOT_DIR/.venv-release-macos/bin/pyinstaller}"
BUILD_DIR="$ROOT_DIR/build/release-macos"
DIST_DIR="$ROOT_DIR/dist/macos"
mkdir -p "$BUILD_DIR/SyLC.iconset" "$DIST_DIR"
for size in 16 32 128 256 512; do
    sips -z "$size" "$size" assets/icon.png \
        --out "$BUILD_DIR/SyLC.iconset/icon_${size}x${size}.png" >/dev/null
done
iconutil -c icns "$BUILD_DIR/SyLC.iconset" -o "$BUILD_DIR/SyLC.icns"
"$PACKAGER" --noconfirm --distpath "$DIST_DIR" \
    --workpath "$ROOT_DIR/build/pyinstaller-macos" scripts/SyLC-macos.spec
APP="$DIST_DIR/SyLC 3D Player.app"
codesign --verify --deep --strict "$APP"
STAGE_DIR="$(mktemp -d "$BUILD_DIR/dmg.XXXXXX")"
ditto "$APP" "$STAGE_DIR/SyLC 3D Player.app"
ln -s /Applications "$STAGE_DIR/Applications"
cat > "$STAGE_DIR/LEEME.txt" <<'EOF'
SyLC 3D Player — macOS Apple Silicon

Arrastra SyLC 3D Player a Applications.
Incluye Python, libmpv, FFmpeg y el modelo Small 518 para 2D→3D.
Esta compilación tiene firma local y no está notarizada por Apple.
Si macOS bloquea la apertura, usa Ajustes del Sistema > Privacidad y seguridad
> Abrir igualmente, después de intentar abrir la aplicación.

La inferencia 2D→3D en este Mac produce aproximadamente 0,8 mapas/s.
No se ha ejecutado el reproductor como prueba de este paquete.
EOF
DMG="$DIST_DIR/SyLC-7.0.0-macOS-arm64.dmg"
hdiutil create -ov -volname 'SyLC 3D Player' -srcfolder "$STAGE_DIR" \
    -format UDZO "$DMG"
hdiutil verify "$DMG"
shasum -a 256 "$DMG" > "$DMG.sha256"
echo "Release: $DMG"
