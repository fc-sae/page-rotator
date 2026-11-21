"""Servidor local de etiquetado: sirve la interfaz, lotes de páginas, miniaturas y guarda etiquetas.

Solo escucha en 127.0.0.1. Uso: uv run python label.py [--port 8765] [--target 1000] [--no-browser]
Con --predictions (salida de predict.py) la cola va de menor a mayor confianza y cada página
arranca con la rotación predicha.
"""

from __future__ import annotations

import argparse
import csv
import errno
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from labels import LabelStore, build_queue, image_path, load_predictions

REPO = Path(__file__).resolve().parent.parent
UI_PATH = Path(__file__).resolve().parent / "label_ui.html"
MAX_BODY = 1024 * 1024
MAX_BATCH = 200
CONTENT_TYPES = {".jpg": "image/jpeg", ".png": "image/png"}


def _load_clusters(clusters_csv: Path | None) -> dict[tuple[str, int], int]:
    out: dict[tuple[str, int], int] = {}
    if clusters_csv is not None and Path(clusters_csv).exists():
        with Path(clusters_csv).open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[(r["pdf_id"], int(r["page"]))] = int(r["cluster"])
    return out


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, pages_dir, clusters_csv, labels_csv, thumbs_dir, target, predictions=None):
        super().__init__(addr, _Handler)
        self.pages_dir = Path(pages_dir)
        self.thumbs_dir = Path(thumbs_dir)
        self.target = target
        self.store = LabelStore(labels_csv)
        self.lock = threading.Lock()
        self.cluster_of = _load_clusters(clusters_csv)
        self.preds: dict[tuple[str, int], dict] | None = None
        if predictions is not None:
            self.queue, self.preds = load_predictions(predictions)
        else:
            self.queue = build_queue(self.pages_dir / "manifest.csv", clusters_csv)

    def progress(self) -> dict:
        skipped = sum(1 for r in self.store.labels.values() if r["status"] == "skip")
        return {"labeled": len(self.store), "target": self.target, "total": len(self.queue),
                "skipped": skipped}


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, format, *args):  # silencia el log por petición
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _text404(self, msg: str = "No encontrado") -> None:
        self._send(404, msg.encode("utf-8"), "text/plain; charset=utf-8")

    def _host_ok(self) -> bool:
        port = self.server.server_address[1]
        return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

    def do_GET(self) -> None:
        if not self._host_ok():
            return self._send(403, "Host no permitido".encode("utf-8"), "text/plain; charset=utf-8")
        url = urlsplit(self.path)
        if url.path == "/":
            try:
                body = UI_PATH.read_bytes()
            except OSError:
                return self._text404("Falta label_ui.html")
            return self._send(200, body, "text/html; charset=utf-8")
        if url.path == "/api/batch":
            return self._batch(parse_qs(url.query))
        if url.path.startswith("/img/"):
            return self._img(url.path[len("/img/"):], parse_qs(url.query))
        self._text404()

    def _batch(self, qs: dict) -> None:
        try:
            n = int(qs.get("n", ["24"])[0])
        except ValueError:
            n = 24
        n = max(0, min(n, MAX_BATCH))
        srv = self.server
        with srv.lock:
            pages = []
            for key in srv.queue:
                if len(pages) >= n:
                    break
                if key not in srv.store:
                    item = {"pdf_id": key[0], "page": key[1], "cluster": srv.cluster_of.get(key)}
                    if srv.preds is not None:
                        item.update(srv.preds[key])
                    pages.append(item)
            progress = srv.progress()
        self._json(200, {"pages": pages, "progress": progress})

    def _img(self, rest: str, qs: dict) -> None:
        parts = rest.split("/")
        if len(parts) != 2:
            return self._text404()
        try:
            page = int(parts[1])
        except ValueError:
            return self._text404()
        size = qs.get("size", ["thumb"])[0]
        try:
            path = image_path(self.server.pages_dir, self.server.thumbs_dir, parts[0], page, size)
            data = path.read_bytes() if path is not None else None
        except Exception:  # PNG corrupto, disco, etc.: no tumbar el manejador
            data = None
        if data is None:
            return self._text404()
        self._send(200, data, CONTENT_TYPES[path.suffix], {"Cache-Control": "max-age=86400"})

    def do_POST(self) -> None:
        if not self._host_ok():
            return self._json(403, {"error": "Host no permitido"})
        if urlsplit(self.path).path != "/api/labels":
            return self._text404()
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            return self._json(415, {"error": "se requiere Content-Type: application/json"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._json(400, {"error": "Content-Length inválido"})
        if length > MAX_BODY:
            return self._json(413, {"error": "cuerpo demasiado grande"})
        raw = self.rfile.read(length)
        try:
            rows = json.loads(raw)
            if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
                raise ValueError("se esperaba una lista de objetos")
            with self.server.lock:
                self.server.store.add(rows)
                progress = self.server.progress()
        except (ValueError, KeyError, TypeError) as e:
            return self._json(400, {"error": f"petición inválida: {e!r}"})
        except OSError as e:
            return self._json(500, {"error": f"No se pudo escribir labels.csv: {e}"})
        self._json(200, {"progress": progress})


def make_server(pages_dir: Path, clusters_csv: Path | None, labels_csv: Path, thumbs_dir: Path,
                target: int = 1000, port: int = 0, predictions: Path | None = None) -> ThreadingHTTPServer:
    """Crea el servidor en 127.0.0.1 (port=0 elige un puerto libre); no lo arranca."""
    return _Server(("127.0.0.1", port), pages_dir, clusters_csv, labels_csv, thumbs_dir, target, predictions)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Herramienta local de etiquetado de rotación.")
    ap.add_argument("--pages", type=Path, default=REPO / "data" / "pages")
    ap.add_argument("--clusters", type=Path, default=REPO / "data" / "clusters" / "clusters.csv")
    ap.add_argument("--labels", type=Path, default=REPO / "data" / "labels.csv")
    ap.add_argument("--thumbs", type=Path, default=REPO / "data" / "thumbs")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--predictions", type=Path, default=None,
                    help="predictions.csv de predict.py: revisar primero las de menor confianza")
    args = ap.parse_args(argv)

    clusters = args.clusters if args.clusters.exists() else None
    manifest = args.pages / "manifest.csv"
    if not manifest.exists():
        sys.exit(f"No existe {manifest}; corre render.py primero.")
    if args.predictions is not None and not args.predictions.exists():
        sys.exit(f"No existe {args.predictions}; corre predict.py primero.")
    try:
        srv = make_server(args.pages, clusters, args.labels, args.thumbs, args.target, args.port,
                          args.predictions)
    except OSError as e:
        sys.exit(f"El puerto {args.port} está ocupado (¿ya está abierta la herramienta?). Usa --port."
                 if e.errno == errno.EADDRINUSE else f"No se pudo iniciar el servidor: {e}")
    except ValueError as e:
        sys.exit(str(e))
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    p = srv.progress()
    print(f"Etiquetado en {url}  ({p['labeled']}/{p['target']} etiquetadas, {p['total']} páginas en cola)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nCerrando.")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
