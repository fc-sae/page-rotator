# page-rotator — Diseño y plan

## Objetivo
Detectar la orientación (0/90/180/270°) de cada página de constancias de idioma en PDF y corregirla
con una CLI en Rust que reemplace a `ocrmypdf --rotate-pages` en el script de sellado.

**Validación:** sobre un conjunto de prueba revisado a mano, `page-rotator` debe tener **menos errores**
de orientación y **menor tiempo** por PDF que `ocrmypdf -q --rotate-pages --rotate-pages-threshold 0.9 --skip-text`.

## Contexto y restricciones
- ~5000 PDFs, ~24 tipos de constancia, distribución desbalanceada.
- Un PDF puede tener varias páginas con orientaciones distintas y varios tipos.
- Origen: Google Drive, lista de URLs (cada archivo en la carpeta de expediente del alumno). Descarga con `rclone`.
- Entrenamiento local: RTX 4060 8 GB. Inferencia en CPU (sin Python ni GPU en producción).
- El OCR no se necesita: solo se rota.

## Arquitectura
```
tools/   (Python, uv)   descarga → rasterizado → etiquetado → entrenamiento → export ONNX → evaluación
rotator/ (Rust)         CLI: PDF → render página → ONNX → ajusta /Rotate → PDF
data/    (gitignored)   raw/, pages/, labels.csv, splits/, models/
```

### Modelo
- Clasificador de 4 clases, backbone preentrenado ligero (MobileNetV3-Large o EfficientNet-B0, `timm`).
- Entrada: página en gris, lado largo 384 px, letterbox a 384×384.
- **Etiquetas sintéticas:** solo se etiqueta la página *derecha*; el entrenamiento la rota a las 4 clases.
- Aumentos: inclinación ±5°, recorte, desenfoque, JPEG, contraste, ruido de escaneo.
- Desbalance: muestreo ponderado por tipo (peso ∝ 1/√n_tipo).
- Partición **por PDF** y estratificada por tipo: 80/10/10 (train/val/test).
- Export a ONNX (opset 17); se verifica igualdad de salida PyTorch vs ONNX.

### CLI `page-rotator`
```
page-rotator [-q] [--threshold 0.9] [--model PATH] [--dry-run] [--json] IN.pdf OUT.pdf
```
- Crates: `clap`, `lopdf` (lee/escribe `/Rotate`), `pdfium-render` (rasteriza), `ort` (ONNX), `anyhow`.
- Por página: render → predicción en imagen y en imagen rotada 180°. Se corrige solo si ambas son
  consistentes y la confianza ≥ `threshold`; si no, la página queda intacta.
- `nuevo /Rotate = (actual + corrección) mod 360`. No se re-rasteriza ni se pierde calidad.
- Modelo embebido en el binario por defecto (`--model` para sobrescribir).
- Un hilo por proceso (el script corre en paralelo con `export -f`).
- Códigos de salida: 0 OK, 1 error de PDF/modelo, 2 argumentos inválidos. `--json` reporta por página.
- Dependencia en producción: `libpdfium.so` junto al binario.

### Integración (reemplazo en el script)
```bash
page-rotator -q --threshold 0.9 "temp_$1" "$1" || { _error "page-rotator falló: $1"; rm -f "temp_$1"; return 1; }
```

## Etapas y responsables

| # | Etapa | Claude | Héctor |
|---|-------|--------|--------|
| 0 | Setup | Estructura del repo, entorno `uv` + PyTorch CUDA, crate Rust | `rclone config` (remote `gdrive`), entregar archivo de URLs |
| 1 | Descarga | `download.py`: URL→ID→`rclone backend copyid`, `qpdf --decrypt`, manifiesto CSV, reintentos | Ejecutarlo / autorizar acceso |
| 2 | Rasterizado | `render.py`: páginas a PNG 150 dpi + manifiesto (pdf, página, tamaño) | — |
| 3a | Tipos | `cluster.py`: embeddings + k-means (k≈30), hoja de contacto por clúster | Nombrar/fusionar clústeres en los ~24 tipos |
| 3b | Etiquetado | Herramienta web local: cuadrícula de miniaturas, clic/tecla rota, guarda `labels.csv` | Etiquetar ~1000 páginas iniciales |
| 3c | Bootstrap | Entrenar modelo inicial, predecir el resto, cola ordenada por baja confianza | Revisar solo la cola de dudosas |
| 3d | Test dorado | Seleccionar test (10% por PDF, estratificado) + fallos conocidos de ocrmypdf | Revisar al 100% el test |
| 4 | Entrenamiento | `train.py` (AMP, AdamW, ~15 épocas, métricas por clase y por tipo), `export.py` | — |
| 5 | CLI Rust | `rotator/` con tests (PDF sintético rotado, multi-página, mixto, cifrado, vacío) | — |
| 6 | Validación | `bench.sh`: errores y tiempo vs ocrmypdf en el test, reporte | Aprobar resultados |
| 7 | Integración | Instrucciones de despliegue (binario + libpdfium) | Cambiar la línea en el script y probar en producción |

## Entregables
1. Este plan.
2. `tools/` — programa de descarga, etiquetado y entrenamiento.
3. `data/models/orientation.onnx` — modelo (versionado fuera de git o con Git LFS).
4. `rotator/` — binario `page-rotator`.

## Riesgos
- **Etiquetas erróneas en test** → test revisado al 100% por Héctor.
- **Páginas en blanco o casi vacías** → baja confianza ⇒ no se tocan.
- **PDFs cifrados** → el script ya aplica `qpdf --decrypt` antes; la CLI falla con código 1 si no puede abrir.
- **Tipos nuevos en el futuro** → el modelo aprende rasgos generales (texto); se mide error por tipo y
  se reentrena con páginas nuevas si aparece degradación.
