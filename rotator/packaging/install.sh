#!/usr/bin/env bash
# Instala page-rotator desde el paquete (directorio donde está este script).
#
#   ./install.sh                  instala en /usr/local (root) o en ~/.local (usuario)
#   ./install.sh --prefix DIR     instala en DIR
#   ./install.sh --uninstall      desinstala (con el mismo --prefix si se usó uno)
#
# Deja:
#   PREFIX/lib/page-rotator/page-rotator    binario (con el modelo incrustado)
#   PREFIX/lib/page-rotator/libpdfium.so    lo busca junto al binario real
#   PREFIX/bin/page-rotator                 enlace simbólico al binario
#   PREFIX/share/bash-completion/completions/page-rotator
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $EUID -eq 0 ]]; then prefix=/usr/local; else prefix="$HOME/.local"; fi
uninstall=0

usage() { sed -n '2,13s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"; }
die() { echo "install.sh: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix) [[ $# -ge 2 ]] || die "--prefix necesita un directorio"; prefix="$2"; shift 2 ;;
        --prefix=*) prefix="${1#--prefix=}"; shift ;;
        --uninstall) uninstall=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ "$prefix" = /* ]] || prefix="$(pwd)/$prefix"

libdir="$prefix/lib/page-rotator"
bin="$prefix/bin/page-rotator"
comp="$prefix/share/bash-completion/completions/page-rotator"

if [[ $uninstall -eq 1 ]]; then
    rm -f "$bin" "$comp" "$libdir/page-rotator" "$libdir/libpdfium.so"
    rmdir "$libdir" 2>/dev/null || true
    echo "page-rotator desinstalado de $prefix"
    exit 0
fi

for f in page-rotator libpdfium.so page-rotator.bash smoke.pdf; do
    [[ -f "$here/$f" ]] || die "falta $f junto a install.sh (¿paquete incompleto?)"
done

# --- requisitos del sistema ------------------------------------------------------------------
[[ "$(uname -s)-$(uname -m)" == "Linux-x86_64" ]] || die "este paquete es para Linux x86_64 (esto es $(uname -s) $(uname -m))"

ver_ge() { [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" == "$2" ]]; }
need_glibc="$(cat "$here/GLIBC" 2>/dev/null || echo 2.38)"
have_glibc="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
if [[ -n "$have_glibc" ]] && ! ver_ge "$have_glibc" "$need_glibc"; then
    die "se necesita glibc >= $need_glibc y el sistema tiene $have_glibc (Ubuntu 24.04+ / Debian 13+)"
fi
missing="$(ldd "$here/page-rotator" 2>&1 | grep -E 'not found|no se encontró|version .* not found' || true)"
[[ -z "$missing" ]] || die "faltan bibliotecas del sistema para el binario:
$missing
(instala libstdc++6 reciente: GCC 13 o posterior)"

# --- copia -----------------------------------------------------------------------------------
mkdir -p "$libdir" "$(dirname "$bin")" "$(dirname "$comp")"
install -m 0755 "$here/page-rotator" "$libdir/page-rotator.new"
install -m 0644 "$here/libpdfium.so" "$libdir/libpdfium.so.new"
mv -f "$libdir/page-rotator.new" "$libdir/page-rotator"   # mv atómico: no rompe procesos en curso
mv -f "$libdir/libpdfium.so.new" "$libdir/libpdfium.so"
ln -sfn "$libdir/page-rotator" "$bin"
install -m 0644 "$here/page-rotator.bash" "$comp"

# --- prueba de humo: binario, modelo incrustado y libpdfium ------------------------------------
if ! out="$(env -u PDFIUM_LIB "$bin" --dry-run --json "$here/smoke.pdf" 2>&1)"; then
    die "la prueba de humo falló:
$out"
fi
[[ "$out" == *'"changed":0'* ]] || die "salida inesperada en la prueba de humo: $out"

echo "page-rotator $("$bin" --version | awk '{print $2}') instalado en $prefix"
echo "  binario:    $bin -> $libdir/page-rotator"
echo "  completion: $comp"
case ":$PATH:" in
    *":$prefix/bin:"*) ;;
    *) echo "  aviso: $prefix/bin no está en PATH; agrégalo (p. ej. en ~/.bashrc: export PATH=\"$prefix/bin:\$PATH\")" ;;
esac
echo "  la completion se activa en shells nuevos (requiere el paquete bash-completion)"
