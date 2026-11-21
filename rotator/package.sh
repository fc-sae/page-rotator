#!/usr/bin/env bash
# Arma el paquete de instalación de page-rotator:
#   dist/page-rotator-VERSION-linux-x86_64.tar.gz  (+ .sha256)
# con el binario release (modelo incrustado), libpdfium.so, la completion de bash, install.sh y un PDF
# para la prueba de humo. En el servidor: tar xzf …; cd page-rotator-…; ./install.sh
#
#   ./package.sh                    usa data/models/orientation.onnx y la libpdfium de tools/.venv
#   PAGE_ROTATOR_MODEL=… PDFIUM_LIB=… ./package.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
die() { echo "package.sh: $*" >&2; exit 1; }

model="${PAGE_ROTATOR_MODEL:-../data/models/orientation.onnx}"
[[ -f "$model" ]] || die "no existe el modelo $model (corre tools/export.py o define PAGE_ROTATOR_MODEL)"
pdfium="${PDFIUM_LIB:-$(ls ../tools/.venv/lib/python*/site-packages/pypdfium2_raw/libpdfium.so 2>/dev/null | head -1)}"
[[ -f "$pdfium" ]] || die "no encontré libpdfium.so (corre 'uv sync' en tools/ o define PDFIUM_LIB)"

PAGE_ROTATOR_MODEL="$(realpath "$model")" cargo build --release --locked
bin=target/release/page-rotator
version="$("$bin" --version | awk '{print $2}')"
name="page-rotator-$version-linux-x86_64"
stage="dist/$name"
rm -rf "$stage"
mkdir -p "$stage"

cp "$bin" "$stage/page-rotator"
strip "$stage/page-rotator" 2>/dev/null || true
cp "$pdfium" "$stage/libpdfium.so"
cp packaging/install.sh packaging/smoke.pdf "$stage/"
chmod +x "$stage/install.sh"
"$stage/page-rotator" --completions bash > "$stage/page-rotator.bash"
# versión mínima de glibc que pide el binario (install.sh la revisa antes de copiar)
objdump -T "$stage/page-rotator" | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -uV | tail -1 > "$stage/GLIBC"
sha256sum "$model" | awk '{print $1 "  orientation.onnx"}' > "$stage/MODEL.sha256"

# el binario del paquete debe funcionar sin --model y con la libpdfium del paquete
out="$(env -u PDFIUM_LIB "$stage/page-rotator" --pdfium "$stage/libpdfium.so" --dry-run --json packaging/smoke.pdf 2>&1)" \
    || die "el binario empaquetado no pasó la prueba de humo: $out"

tar -C dist -czf "dist/$name.tar.gz" "$name"
(cd dist && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")
rm -rf "$stage"
echo "dist/$name.tar.gz  ($(du -h "dist/$name.tar.gz" | cut -f1), glibc >= $(tar -xzOf "dist/$name.tar.gz" "$name/GLIBC"))"
