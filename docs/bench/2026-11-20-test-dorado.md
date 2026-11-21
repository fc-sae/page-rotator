# Etapa 6 — page-rotator vs ocrmypdf en el test dorado (2026-11-20)

**Conjunto:** partición `test` (10 % de los PDFs, por PDF y estratificada por tipo): 443 PDFs, 513 páginas,
revisadas al 100 % por Héctor. Etiquetas: 435 derechas, 51 a 90°, 26 a 270°, 1 a 180°.

**Comandos** (uno tras otro, por PDF, en la misma máquina):
- `page-rotator -q --threshold 0.9 IN OUT` (modelo v3: 5002 etiquetas, MobileNetV3-Large, ONNX embebido)
- `ocrmypdf -q --rotate-pages --rotate-pages-threshold 0.9 --skip-text IN OUT` (línea de producción, ocrmypdf 15.2.0)

**Medición:** el giro aplicado se mide visualmente comparando la página de entrada con la de salida,
porque ocrmypdf reescribe el PDF y gira el contenido en lugar de `/Rotate`. En page-rotator la medición
visual coincide en 513/513 páginas con la decisión de la CLI.

| herramienta | bien | giro equivocado | sin corregir | PDFs con error | reloj mediana | reloj p95 | reloj total | CPU total |
|---|---|---|---|---|---|---|---|---|
| page-rotator | **513 (100 %)** | **0** | **0** | 0 | **250 ms** | **436 ms** | **116 s** | **118 s** |
| ocrmypdf | 492 (95.9 %) | 10 (1.9 %) | 11 (2.1 %) | 2 | 2692 ms | 9142 ms | 1442 s | 1841 s |

- page-rotator es ~11× más rápido por PDF (mediana) y usa ~16× menos CPU en total.
- Errores de ocrmypdf: 8 páginas derechas que volteó 180°, 2 páginas a 270° giradas al revés (90°),
  11 páginas giradas que no corrigió (6 a 270°, 4 a 90°, 1 a 180°).
- Los 2 PDFs con error de ocrmypdf tienen firma digital (código 2, se niega a invalidarla). page-rotator
  los procesa: si no hay correcciones copia los bytes tal cual (firma intacta); si las hubiera, la
  actualización incremental conserva la revisión firmada, pero los visores mostrarán que el documento
  cambió después de firmarse.
- Las 5 páginas de test en las que el modelo no coincide con la etiqueta (confianza 0.39–0.73) fueron
  inconsistentes entre la vista directa y la girada 180°, así que la regla de decisión no las tocó.

**Resultado:** cumple los dos criterios del diseño (menos errores y menor tiempo que ocrmypdf).

Reproducir: `cd tools && uv run python bench.py` (escribe `data/bench/report.md`, `pages.csv`, `pdfs.csv`).
