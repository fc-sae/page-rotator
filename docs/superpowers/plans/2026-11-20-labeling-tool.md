# Herramienta de etiquetado (etapa 3b) — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `uv run python label.py` abre en el navegador una cuadrícula de 24 páginas para marcar su rotación y guarda `data/labels.csv`.

**Architecture:** Lógica pura (etiquetas, cola, miniaturas) en `tools/labels.py`; servidor HTTP de la biblioteca estándar en `tools/label.py`; interfaz en un solo archivo `tools/label_ui.html` (HTML + CSS + JS sin dependencias). El servidor guarda estado en memoria (cola y etiquetas) protegido con un lock y persiste cada pantalla en `labels.csv`.

**Tech Stack:** Python 3.12 (`http.server`, `csv`, `json`), Pillow, pytest; JS sin frameworks.

**Spec:** `docs/superpowers/specs/2026-11-20-labeling-tool-design.md`

## Global Constraints

- Sin dependencias nuevas en `tools/pyproject.toml`.
- Servidor solo en `127.0.0.1`.
- `labels.csv` columnas exactas: `pdf_id,page,rotation,status,labeled_at`; `rotation` ∈ {0,90,180,270} en sentido horario sobre la página tal como se ve; `status` ∈ {`ok`,`skip`}; solo agregar filas; la última fila por `(pdf_id, page)` gana.
- 24 miniaturas por pantalla; miniatura = lado largo 320 px JPEG en `data/thumbs/`.
- Teclas: ←→↑↓ mover · R ↷ (+90) · E ↶ (−90) · X duda · Espacio ver grande · Enter guardar y siguiente · Retroceso pantalla anterior.
- Insignias: `↷ 90°` (90), `↻ 180°` (180), `↶ 90°` (270), `? duda` (skip); sin insignia en 0.
- Textos de la interfaz en español.

## Review Focus

1. Reabrir la herramienta con `labels.csv` existente → la cola salta lo ya etiquetado y el contador arranca donde iba (Task 1 `test_queue_skips_labeled`, Task 3 `test_progress_counts_existing_labels`).
2. Rutas maliciosas o raras (`/img/../../etc/passwd/1`, página no numérica, página inexistente) → 400/404, nunca un archivo fuera de `data/` (Task 2 `test_image_rejects_bad_ids`, Task 3 `test_img_rejects_traversal`).
3. Enter dos veces o dos pestañas → filas repetidas, pero el CSV nunca se corrompe y la última gana (Task 1 `test_last_row_wins`, Task 3 `test_concurrent_posts`).
4. Página horizontal girada 90° en la cuadrícula → no se encima con sus vecinas (Task 4, verificación manual con una página de `rotate=90`).
5. Cola terminada → la interfaz muestra "Terminado" y Enter no hace nada (Task 3 `test_batch_empty_when_done`, Task 4 verificación manual).

---

### Task 1: Etiquetas y cola (`tools/labels.py`)

**Files:**
- Create: `tools/labels.py`
- Test: `tools/tests/test_labels.py`

**Interfaces:**
- Produces:
  - `FIELDS = ["pdf_id", "page", "rotation", "status", "labeled_at"]`
  - `class LabelStore(path: Path)` con `.labels: dict[tuple[str, int], dict]` (última fila por página), `.add(rows: list[dict]) -> None` (valida, agrega `labeled_at`, escribe y hace flush; lanza `ValueError` si `rotation` o `status` son inválidos), `__contains__(key: tuple[str, int])`, `__len__`.
  - `build_queue(pages_manifest: Path, clusters_csv: Path | None, seed: int = 0) -> list[tuple[str, int]]` — páginas `ok` en orden round-robin por clúster; dentro de cada clúster orden aleatorio fijo (`random.Random(seed)`); clústeres recorridos en orden ascendente; páginas sin clúster forman un grupo final `"sin_cluster"` que también entra en el round-robin. No filtra etiquetadas (eso lo hace el servidor).

- [ ] **Step 1: Escribir pruebas que fallan**
  - `test_add_and_reload`: agrega 2 filas, nuevo `LabelStore` sobre el mismo archivo las lee con `page` int y `rotation` int.
  - `test_last_row_wins`: `add` (`a`,1,90,ok) y luego (`a`,1,0,ok) → `labels[("a",1)]["rotation"] == 0`, `len == 1`, el CSV tiene 2 filas de datos.
  - `test_add_rejects_invalid`: `rotation=45` → `ValueError`; `status="x"` → `ValueError`; el archivo no cambia.
  - `test_round_robin_by_cluster`: manifiesto con 3 páginas en clúster 0, 1 en clúster 1, 1 sin clúster → los 3 primeros elementos de la cola vienen de 3 grupos distintos; total 5; misma semilla → misma cola.
  - `test_queue_without_clusters_file`: `clusters_csv=None` → todas las páginas `ok`, las `error` excluidas.
  - `test_queue_skips_labeled` vive en Task 3 (el filtrado es del servidor).
- [ ] **Step 2:** `cd tools && uv run pytest tests/test_labels.py -v` → FAIL (`ModuleNotFoundError: labels`).
- [ ] **Step 3:** Implementar `tools/labels.py` según Interfaces.
- [ ] **Step 4:** `uv run pytest tests/test_labels.py -v` → PASS.
- [ ] **Step 5:** Commit `Etapa 3b: labels.py, almacén de etiquetas y cola por clúster`.

### Task 2: Miniaturas y validación de rutas (`tools/labels.py`)

**Files:**
- Modify: `tools/labels.py`
- Test: `tools/tests/test_labels.py`

**Interfaces:**
- Produces:
  - `ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,}$")`
  - `image_path(pages_dir: Path, thumbs_dir: Path, pdf_id: str, page: int, size: str) -> Path | None` — `size` ∈ {`"thumb"`, `"full"`}; `None` si `pdf_id` no cumple `ID_RE`, `page < 1`, `size` inválido o el PNG no existe. `"full"` devuelve el PNG original; `"thumb"` genera (si falta) `thumbs_dir/<pdf_id>/<NNN>.jpg` con lado largo 320 px, calidad 80, y lo devuelve.

- [ ] **Step 1: Pruebas que fallan**
  - `test_thumb_created_once`: PNG 1275×1650 → thumb de 247×320; segunda llamada no reescribe (mismo `st_mtime_ns`).
  - `test_full_returns_png`.
  - `test_image_rejects_bad_ids`: `"../../etc"`, `"abc"` (corto), `page=0`, `size="big"`, página inexistente → `None`.
- [ ] **Step 2:** correr → FAIL.
- [ ] **Step 3:** Implementar con Pillow (`thumbnail((320, 320))`, escribir a `.part` y `replace`).
- [ ] **Step 4:** correr → PASS.
- [ ] **Step 5:** Commit `Etapa 3b: miniaturas y validación de rutas`.

### Task 3: Servidor (`tools/label.py`)

**Files:**
- Create: `tools/label.py`
- Test: `tools/tests/test_label_server.py`

**Interfaces:**
- Consumes: `LabelStore`, `build_queue`, `image_path` (Tasks 1–2).
- Produces:
  - `make_server(pages_dir: Path, clusters_csv: Path | None, labels_csv: Path, thumbs_dir: Path, target: int = 1000, port: int = 0) -> ThreadingHTTPServer` (escucha en `127.0.0.1`; `port=0` = puerto libre).
  - `main(argv)` con `--port` (default 8765), `--target` (1000), `--no-browser`; abre `webbrowser` salvo `--no-browser`.
  - Rutas:
    - `GET /` → `label_ui.html`.
    - `GET /api/batch?n=24` → `{"pages": [{"pdf_id", "page", "cluster"}], "progress": {"labeled", "target", "total"}}` con las primeras `n` páginas de la cola que no están en el `LabelStore`.
    - `POST /api/labels` cuerpo `[{"pdf_id", "page", "rotation", "status"}]` → `LabelStore.add`; responde `{"progress": …}`; `400` con `{"error"}` si es inválido.
    - `GET /img/<pdf_id>/<page>?size=thumb|full` → imagen con `Content-Type` correcto y `Cache-Control: max-age=86400`; `404` si `image_path` da `None`.
  - Todo acceso al `LabelStore` bajo un `threading.Lock`.

- [ ] **Step 1: Pruebas que fallan** (servidor en hilo, `urllib.request`, datos sintéticos de `test_render.make_pdf` o PNGs generados con PIL)
  - `test_batch_returns_unlabeled_pages`, `test_post_labels_then_batch_skips_them`.
  - `test_queue_skips_labeled` / `test_progress_counts_existing_labels`: `labels.csv` previo con 2 filas → `progress.labeled == 2` y esas páginas no salen.
  - `test_post_invalid_returns_400` (`rotation=45`).
  - `test_img_thumb_and_full` (content-type `image/jpeg` / `image/png`).
  - `test_img_rejects_traversal`: `/img/..%2F..%2Fetc/1`, `/img/abcdefghijk/x` → 404/400.
  - `test_batch_empty_when_done`: todo etiquetado → `pages == []`.
  - `test_concurrent_posts`: 8 hilos hacen POST de la misma página → CSV con 8 filas válidas, `len(store) == 1`.
  - `test_binds_localhost`: `server.server_address[0] == "127.0.0.1"`.
- [ ] **Step 2:** correr → FAIL.
- [ ] **Step 3:** Implementar con `http.server.BaseHTTPRequestHandler` (silenciar `log_message`).
- [ ] **Step 4:** correr → PASS.
- [ ] **Step 5:** Commit `Etapa 3b: servidor de etiquetado`.

### Task 4: Interfaz (`tools/label_ui.html`)

**Files:**
- Create: `tools/label_ui.html`
- Modify: `README.md` (sección "Etiquetado" con el comando y las teclas)

**Interfaces:**
- Consumes: las rutas de Task 3.

- [ ] **Step 1:** Implementar la página:
  - Barra superior: `Etiquetado — {labeled} / {target}` y ayuda `←→↑↓ mover · R ↷ derecha · E ↶ izquierda · X duda · Espacio ver grande · Enter guardar y siguiente · Retroceso anterior`.
  - Cuadrícula responsiva de celdas cuadradas; la imagen va dentro de un contenedor cuadrado con `transform: rotate(<rotation>deg)` y tamaño ajustado para que una página girada 90° quepa en la celda.
  - Cada celda: botones `↶` y `↷`, insignia según Global Constraints, borde resaltado en la seleccionada, celda atenuada con `? duda`.
  - Estado local por pantalla `{key: {rotation, status}}`, todas empiezan en `0/ok`. X alterna duda.
  - Enter: `POST /api/labels` y luego `GET /api/batch`; guarda la pantalla enviada en un historial. Retroceso: recarga la última pantalla del historial con sus etiquetas para corregir; Enter la vuelve a enviar (la última fila gana).
  - Espacio: superposición con `size=full` y la rotación aplicada; Espacio o Esc la cierra.
  - Cola vacía: mensaje "Terminado: no quedan páginas por etiquetar"; Enter no hace nada.
  - Error de red: mensaje visible, la pantalla no avanza.
- [ ] **Step 2:** Verificación: `uv run python label.py --no-browser --port 8765`, abrir `http://127.0.0.1:8765/` y comprobar a mano: R/E giran con la insignia correcta, una página con `rotate=90` en el manifiesto no se encima, Enter escribe 24 filas en `data/labels.csv`, Retroceso muestra la pantalla anterior con sus etiquetas, recargar la página continúa donde iba.
- [ ] **Step 3:** `uv run pytest -q` → todo PASS.
- [ ] **Step 4:** Commit `Etapa 3b: interfaz de etiquetado`.
