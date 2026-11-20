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

## Etiquetado

Herramienta local para marcar la rotación correcta de las páginas (genera `data/labels.csv`):

```bash
cd tools
uv run python label.py    # abre http://127.0.0.1:8765/ ; --no-browser, --port, --target
```

Cada pantalla muestra 24 páginas ya con la corrección aplicada (todas empiezan derechas).

| Tecla | Acción |
|---|---|
| ← → ↑ ↓ | mover la selección |
| R | girar 90° a la derecha (↷) |
| E | girar 90° a la izquierda (↶) |
| X | marcar o desmarcar duda / en blanco (alterna) |
| Espacio | ver grande (Espacio o Esc cierran) |
| Enter | guardar la pantalla y pasar a la siguiente (espera a que carguen las imágenes) |
| Retroceso | volver a la pantalla anterior para corregir |

Cada celda también tiene botones ↶/↷ para girar con el ratón. Si una miniatura no carga, la celda
queda como duda con la marca `sin imagen` (R/E/X no la afectan) y se guarda como `skip` para salir de la cola.
Enter se ignora ("Cargando imágenes…") hasta que todas las miniaturas terminaron de cargar.
Al volver con Retroceso se conservan las correcciones de la pantalla actual aún no guardada.
El encabezado muestra `etiquetadas / meta (N duda)`.

Las etiquetas se agregan a `data/labels.csv` en cada Enter (la última fila de cada página gana),
así que se puede cerrar y continuar después. Las miniaturas se guardan en `data/thumbs/`.
