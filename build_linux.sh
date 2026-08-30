#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST="${ROOT}/dist"
PKG="${DIST}/V-D-Splitter-linux"

rm -rf "${PKG}"
mkdir -p "${PKG}"

rsync -a "${ROOT}/" "${PKG}/" \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude "build" \
  --exclude "dist" \
  --exclude "payload" \
  --exclude "__pycache__" \
  --exclude "*.spec" \
  --exclude "input" \
  --exclude "output" \
  --exclude "work" \
  --exclude "settings.json"

chmod +x "${PKG}/build_linux.sh" || true
chmod +x "${PKG}/installer/bootstrap_installer_linux.py"

cat > "${PKG}/install.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
python3 installer/bootstrap_installer_linux.py
EOF
chmod +x "${PKG}/install.sh"

cd "${DIST}"
tar -czf "V-D-Splitter-linux.tar.gz" "V-D-Splitter-linux"
echo "Built: ${DIST}/V-D-Splitter-linux.tar.gz"
