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

## Entrenamiento y revisión asistida

```bash
cd tools
uv run python train.py            # data/models/orientation.pt + metrics.json (≈5 min en la RTX 4060)
uv run python predict.py          # data/predictions.csv: predicción y confianza de cada página
uv run python label.py --predictions ../data/predictions.csv            # sin etiqueta, menor confianza primero
uv run python label.py --predictions ../data/predictions.csv --recheck  # etiquetadas donde el modelo no coincide
uv run python export.py           # data/models/orientation.onnx (+ .json) para la CLI
```

Con `--predictions` cada página arranca con la rotación predicha (Enter confirma) y el tooltip muestra la
confianza. Con `--recheck` cada página muestra tu etiqueta actual y el tooltip dice qué predice el modelo.
La partición train/val/test es por PDF y se fija en `data/splits/split.csv` la primera vez; `test` será el
test dorado, así que `train.py --test` se usa poco.

## CLI `page-rotator`

```bash
cd tools && uv run python export.py          # data/models/orientation.onnx
cd ../rotator && cargo build --release       # incrusta el modelo si existe data/models/orientation.onnx
target/release/page-rotator [-q] [--threshold 0.9] [--model M.onnx] [--dry-run] [--json] IN.pdf [OUT.pdf]
```

- Solo cambia `/Rotate` (actualización incremental: los bytes originales quedan intactos); si no hay
  cambios, OUT es copia exacta de IN. OUT puede ser el mismo IN.
- Cada página se evalúa tal como se ve y girada 180°; se corrige solo si ambas coinciden y la confianza
  alcanza `--threshold`.
- Salida: 0 OK, 1 error de PDF/modelo (incluye PDF cifrado: aplicar `qpdf --decrypt` antes),
  2 argumentos inválidos.
- Despliegue: el binario (ONNX Runtime estático, modelo incrustado) + `libpdfium.so` en la misma
  carpeta (o `PDFIUM_LIB=/ruta/libpdfium.so`). Para las pruebas se usa la de `tools/.venv` (pypdfium2).
