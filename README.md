# page-rotator

Detecta la orientación (0/90/180/270°) de cada página de constancias en PDF y la corrige ajustando
`/Rotate`, sin re-rasterizar. Reemplaza a `ocrmypdf --rotate-pages` en el script de sellado.

Diseño y plan: [`docs/superpowers/specs/2026-10-10-page-rotator-design.md`](docs/superpowers/specs/2026-10-10-page-rotator-design.md)

## Estructura

```
tools/     Python (uv): descarga → rasterizado → etiquetado → entrenamiento → export ONNX → evaluación
rotator/   Rust: CLI page-rotator (PDF → render → ONNX → ajusta /Rotate → PDF)
data/      (gitignored) urls.txt, raw/, pages/, labels.csv, splits/, models/
```

## Requisitos

- `rclone` con un remote `gdrive` (solo lectura), `qpdf`
- [`uv`](https://docs.astral.sh/uv/) y Python 3.12
- Rust (cargo) para `rotator/`
- GPU NVIDIA para entrenar (la inferencia en producción es solo CPU)

## Entorno

```bash
cd tools
uv sync                 # instala PyTorch CUDA y dependencias
uv run pytest           # verifica imports y GPU

cd ../rotator
cargo build --release
```
