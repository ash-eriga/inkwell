#!/usr/bin/env sh
# Inkwell installer — downloads the latest prebuilt Linux x86_64 CUDA 12 bundle,
# verifies SHA-256 against SHA256SUMS, and installs atomically to ~/.local.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/ash-eriga/inkwell/master/install.sh | sh
#   INKWELL_VERSION=v0.1.0 sh install.sh
set -eu

REPO="ash-eriga/inkwell"
ASSET_SUFFIX="linux-x86_64-cuda12.tar.gz"
DATA_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/inkwell"
BIN_DIR="$HOME/.local/bin"

die() { echo "inkwell install: error: $*" >&2; exit 1; }

need() {
  command -v "$1" >/dev/null 2>&1 || die "required tool not found: $1"
}

need curl
need tar
need sha256sum

[ "$(uname -s)" = "Linux" ] || die "only Linux is supported in v0.1"
[ "$(uname -m)" = "x86_64" ] || die "only x86_64 is supported in v0.1"

if [ -n "${INKWELL_VERSION:-}" ]; then
  TAG="$INKWELL_VERSION"
  case "$TAG" in v*) ;; *) TAG="v$TAG" ;; esac
else
  TAG="$(curl -fsSL -H 'Accept: application/vnd.github+json' \
    "https://api.github.com/repos/$REPO/releases/latest" \
    | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
    | head -n 1)"
  [ -n "$TAG" ] || die "could not determine latest release tag"
fi

case "$TAG" in
  ""|*[!A-Za-z0-9._-]*) die "unsafe release tag: $TAG" ;;
esac

ASSET="inkwell-${TAG}-${ASSET_SUFFIX}"
BASE_URL="https://github.com/$REPO/releases/download/$TAG"
STAGING="$DATA_ROOT/staging/$TAG"
DEST="$DATA_ROOT/versions/$TAG"

rm -rf "$STAGING"
mkdir -p "$STAGING" "$DATA_ROOT/versions" "$BIN_DIR"

echo "Downloading $ASSET..."
curl -fL# --connect-timeout 15 --max-time 300 -o "$STAGING/$ASSET" "$BASE_URL/$ASSET"
curl -fsSL --connect-timeout 15 --max-time 60 -o "$STAGING/SHA256SUMS" "$BASE_URL/SHA256SUMS"

echo "Verifying SHA-256..."
(
  cd "$STAGING"
  grep -F " $ASSET" SHA256SUMS > SHA256SUMS.one || die "$ASSET missing from SHA256SUMS"
  sha256sum -c SHA256SUMS.one >/dev/null
)

EXTRACT="$STAGING/unpack"
BACKUP="$DATA_ROOT/versions/.${TAG}.old-$$"
mkdir -p "$EXTRACT"
tar -xzf "$STAGING/$ASSET" -C "$EXTRACT"
[ -f "$EXTRACT/inkwell" ] || die "archive missing inkwell binary"
chmod 755 "$EXTRACT/inkwell"

rm -rf "$BACKUP"
if [ -d "$DEST" ]; then
  mv -T "$DEST" "$BACKUP" || die "could not move aside existing install"
fi
if ! mv -T "$EXTRACT" "$DEST"; then
  if [ -d "$BACKUP" ]; then mv -T "$BACKUP" "$DEST"; fi
  die "could not install verified archive"
fi
rm -rf "$BACKUP"

ln -sfn "versions/$TAG" "$DATA_ROOT/.current.tmp"
mv -Tf "$DATA_ROOT/.current.tmp" "$DATA_ROOT/current"

cat > "$BIN_DIR/inkwell.tmp" <<EOF
#!/usr/bin/env sh
set -eu
ROOT="$DATA_ROOT/current"
CUDA_LIB=""
for d in "\${INKWELL_CUDA_HOME:-}/lib64" "\${CUDA_HOME:-}/lib64" "\$HOME/lfm-gtx1060-bench/cuda-12.6/lib64" /usr/local/cuda/lib64 /opt/cuda/lib64; do
  if [ -f "\$d/libcudart.so.12" ]; then CUDA_LIB="\$d"; break; fi
done
export LD_LIBRARY_PATH="\$ROOT/lib\${CUDA_LIB:+:\$CUDA_LIB}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export GGML_PASCAL_GRAPHS="\${GGML_PASCAL_GRAPHS:-1}"
export INKWELL_PREFILL_CHUNK="\${INKWELL_PREFILL_CHUNK:-512}"
exec "\$ROOT/inkwell" "\$@"
EOF
chmod 755 "$BIN_DIR/inkwell.tmp"
mv -f "$BIN_DIR/inkwell.tmp" "$BIN_DIR/inkwell"
rm -rf "$STAGING"

echo ""
echo "Installed inkwell $TAG to $DEST"
echo "Run 'inkwell' to get started, or 'inkwell update' to upgrade later."
case ":${PATH:-}:" in
  *":$BIN_DIR:"*) ;;
  *) echo "Note: add $BIN_DIR to your PATH (e.g. export PATH=\"\$HOME/.local/bin:\$PATH\")." ;;
esac
