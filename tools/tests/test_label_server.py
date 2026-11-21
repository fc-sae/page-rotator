import csv
import json
import threading
import urllib.error
import urllib.request

import pytest
from PIL import Image

import label
from labels import FIELDS

PDF_A = "pdfAAAAAAAA01"
PDF_B = "pdfBBBBBBBB02"
HEADER = "pdf_id,page,status,rotate,width_pt,height_pt,width_px,height_px,path,error"


def make_data(tmp_path, pages=((PDF_A, 1), (PDF_A, 2), (PDF_B, 1), (PDF_B, 2))):
    pages_dir = tmp_path / "pages"
    lines = [HEADER]
    for pdf, page in pages:
        d = pages_dir / pdf
        d.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (200, 260), "white").save(d / f"{page:03d}.png")
        lines.append(f"{pdf},{page},ok,0,612,792,200,260,{pdf}/{page:03d}.png,")
    (pages_dir / "manifest.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pages_dir


def start(tmp_path, target=1000, clusters=None):
    pages_dir = make_data(tmp_path) if not (tmp_path / "pages").exists() else tmp_path / "pages"
    srv = label.make_server(pages_dir, clusters, tmp_path / "labels.csv", tmp_path / "thumbs",
                            target=target, port=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, t


@pytest.fixture
def server(tmp_path):
    srv, t = start(tmp_path)
    yield srv
    srv.shutdown()
    srv.server_close()
    t.join()


def url(srv, path):
    return f"http://127.0.0.1:{srv.server_address[1]}{path}"


def get_json(srv, path):
    with urllib.request.urlopen(url(srv, path)) as r:
        assert r.headers["Content-Type"] == "application/json; charset=utf-8"
        return json.load(r)


def post(srv, body, raw=False):
    data = body if raw else json.dumps(body).encode()
    req = urllib.request.Request(url(srv, "/api/labels"), data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req)


def row(pdf=PDF_A, page=1, rotation=90, status="ok"):
    return {"pdf_id": pdf, "page": page, "rotation": rotation, "status": status}


def test_batch_returns_unlabeled_pages(server):
    d = get_json(server, "/api/batch?n=3")
    assert len(d["pages"]) == 3
    assert set(d["pages"][0]) == {"pdf_id", "page", "cluster"}
    assert d["pages"][0]["cluster"] is None
    assert d["progress"] == {"labeled": 0, "target": 1000, "total": 4, "skipped": 0}


def test_post_labels_then_batch_skips_them(server):
    with post(server, [row(PDF_A, 1)]) as r:
        assert json.load(r)["progress"]["labeled"] == 1
    d = get_json(server, "/api/batch?n=24")
    keys = {(p["pdf_id"], p["page"]) for p in d["pages"]}
    assert (PDF_A, 1) not in keys and len(keys) == 3
    assert d["progress"]["labeled"] == 1


def test_progress_counts_existing_labels(tmp_path):
    make_data(tmp_path)
    with (tmp_path / "labels.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerow([PDF_A, 1, 0, "ok", "2026-01-01T00:00:00"])
        w.writerow([PDF_B, 2, 90, "skip", "2026-01-01T00:00:00"])
    srv, t = start(tmp_path)
    try:
        d = get_json(srv, "/api/batch")
        assert d["progress"]["labeled"] == 2
        keys = {(p["pdf_id"], p["page"]) for p in d["pages"]}
        assert keys == {(PDF_A, 2), (PDF_B, 1)}
    finally:
        srv.shutdown()
        srv.server_close()
        t.join()


def test_cluster_from_csv(tmp_path):
    make_data(tmp_path)
    cl = tmp_path / "clusters.csv"
    cl.write_text(f"pdf_id,page,cluster\n{PDF_A},1,3\n", encoding="utf-8")
    srv, t = start(tmp_path, clusters=cl)
    try:
        d = get_json(srv, "/api/batch?n=24")
        by = {(p["pdf_id"], p["page"]): p["cluster"] for p in d["pages"]}
        assert by[(PDF_A, 1)] == 3 and by[(PDF_A, 2)] is None
    finally:
        srv.shutdown()
        srv.server_close()
        t.join()


def test_post_invalid_returns_400(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        post(server, [row(rotation=45)])
    assert e.value.code == 400
    assert "error" in json.load(e.value)
    with pytest.raises(urllib.error.HTTPError) as e:
        post(server, b"no es json", raw=True)
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        post(server, [{"pdf_id": PDF_A}])
    assert e.value.code == 400
    assert get_json(server, "/api/batch")["progress"]["labeled"] == 0


def test_post_too_large_returns_413(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        post(server, b"[" + b" " * (1024 * 1024 + 10) + b"]", raw=True)
    assert e.value.code == 413


def test_img_thumb_and_full(server):
    with urllib.request.urlopen(url(server, f"/img/{PDF_A}/1?size=thumb")) as r:
        assert r.headers["Content-Type"] == "image/jpeg"
        assert r.headers["Cache-Control"] == "max-age=86400"
        assert r.read()[:2] == b"\xff\xd8"
    with urllib.request.urlopen(url(server, f"/img/{PDF_A}/1?size=full")) as r:
        assert r.headers["Content-Type"] == "image/png"
        assert r.read()[:4] == b"\x89PNG"


def test_img_rejects_traversal(server):
    for p in ("/img/..%2F..%2Fetc/1", "/img/abcdefghijk/x", f"/img/{PDF_A}/99",
              f"/img/{PDF_A}/1?size=big", "/nada"):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(url(server, p))
        assert e.value.code in (400, 404), p


def test_img_corrupt_png_is_404(tmp_path):
    pages_dir = make_data(tmp_path)
    (pages_dir / PDF_A / "001.png").write_bytes(b"basura")
    srv, t = start(tmp_path)
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(url(srv, f"/img/{PDF_A}/1?size=thumb"))
        assert e.value.code == 404
        with urllib.request.urlopen(url(srv, f"/img/{PDF_A}/2?size=thumb")) as r:
            assert r.status == 200
    finally:
        srv.shutdown()
        srv.server_close()
        t.join()


def test_batch_empty_when_done(server):
    body = [row(p, n) for p in (PDF_A, PDF_B) for n in (1, 2)]
    with post(server, body):
        pass
    d = get_json(server, "/api/batch")
    assert d["pages"] == [] and d["progress"]["labeled"] == 4


def test_concurrent_posts(server, tmp_path):
    errors = []

    def work():
        try:
            with post(server, [row(PDF_A, 1)]):
                pass
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=work) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    with (tmp_path / "labels.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8 and all(r["rotation"] == "90" and r["status"] == "ok" for r in rows)
    assert get_json(server, "/api/batch")["progress"]["labeled"] == 1


def test_binds_localhost(server):
    assert server.server_address[0] == "127.0.0.1"


def test_index_serves_ui(server):
    with urllib.request.urlopen(url(server, "/")) as r:
        assert r.status == 200
        assert r.headers["Content-Type"].startswith("text/html")
        body = r.read().decode("utf-8")
    assert "↶" in body and "↷" in body


def test_write_oserror_returns_500(server, monkeypatch):
    def boom(rows):
        raise OSError("disco lleno")
    monkeypatch.setattr(server.store, "add", boom)
    with pytest.raises(urllib.error.HTTPError) as e:
        post(server, [row()])
    assert e.value.code == 500
    assert "No se pudo escribir labels.csv" in json.load(e.value)["error"]


def test_wrong_host_is_403(server):
    for path in ("/api/batch", "/"):
        req = urllib.request.Request(url(server, path), headers={"Host": "evil.example:80"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 403
    req = urllib.request.Request(url(server, "/api/labels"), data=b"[]", method="POST",
                                 headers={"Content-Type": "application/json", "Host": "evil.example"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 403
    port = server.server_address[1]
    req = urllib.request.Request(url(server, "/api/batch"), headers={"Host": f"localhost:{port}"})
    with urllib.request.urlopen(req) as r:
        assert r.status == 200


def test_post_requires_json_content_type(server):
    req = urllib.request.Request(url(server, "/api/labels"), data=json.dumps([row()]).encode(),
                                 method="POST", headers={"Content-Type": "text/plain"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 415
    assert get_json(server, "/api/batch")["progress"]["labeled"] == 0


def test_post_non_list_is_400(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        post(server, {"pdf_id": PDF_A})
    assert e.value.code == 400


def test_progress_includes_skipped(server):
    with post(server, [row(PDF_A, 1, 0, "skip"), row(PDF_A, 2, 90, "ok")]) as r:
        p = json.load(r)["progress"]
    assert p["labeled"] == 2 and p["skipped"] == 1
    assert get_json(server, "/api/batch")["progress"]["skipped"] == 1


def test_predictions_mode_orders_by_confidence(tmp_path):
    make_data(tmp_path)
    pred = tmp_path / "pred.csv"
    pred.write_text("pdf_id,page,rotation,confidence,consistent,decided,label,split\n"
                    f"{PDF_A},1,0,0.99,1,0,,train\n{PDF_A},2,90,0.40,1,0,,train\n"
                    f"{PDF_B},1,270,0.70,1,0,,val\n{PDF_B},2,0,0.95,1,0,,test\n")
    srv = label.make_server(tmp_path / "pages", None, tmp_path / "labels.csv", tmp_path / "thumbs",
                            port=0, predictions=pred)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        d = get_json(srv, "/api/batch?n=24")
        assert [(p["pdf_id"], p["page"], p["rotation"]) for p in d["pages"]] == [
            (PDF_A, 2, 90), (PDF_B, 1, 270), (PDF_B, 2, 0), (PDF_A, 1, 0)]
        assert d["pages"][0]["confidence"] == 0.4
        post(srv, [row(PDF_A, 2, 90)]).close()
        assert get_json(srv, "/api/batch?n=1")["pages"][0]["pdf_id"] == PDF_B
    finally:
        srv.shutdown()
        srv.server_close()
        t.join()


def test_predictions_mode_rejects_bad_rotation(tmp_path):
    make_data(tmp_path)
    pred = tmp_path / "pred.csv"
    pred.write_text(f"pdf_id,page,rotation,confidence\n{PDF_A},1,45,0.5\n")
    with pytest.raises(ValueError):
        label.make_server(tmp_path / "pages", None, tmp_path / "labels.csv", tmp_path / "thumbs",
                          port=0, predictions=pred)


def test_recheck_mode_shows_disagreements_with_current_label(tmp_path):
    make_data(tmp_path)
    (tmp_path / "labels.csv").write_text(
        "pdf_id,page,rotation,status,labeled_at\n"
        f"{PDF_A},1,90,ok,t\n{PDF_A},2,0,ok,t\n{PDF_B},1,0,skip,t\n")
    pred = tmp_path / "pred.csv"
    pred.write_text("pdf_id,page,rotation,confidence,consistent,decided,label,split\n"
                    f"{PDF_A},1,0,0.90,1,0,90,train\n{PDF_A},2,0,0.95,1,0,0,train\n"
                    f"{PDF_B},1,90,0.70,1,0,skip,val\n{PDF_B},2,90,0.6,1,0,,test\n")
    srv = label.make_server(tmp_path / "pages", None, tmp_path / "labels.csv", tmp_path / "thumbs",
                            port=0, predictions=pred, recheck=True)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        d = get_json(srv, "/api/batch?n=24")
        assert [(p["pdf_id"], p["page"], p["rotation"], p["model_rotation"]) for p in d["pages"]] == [
            (PDF_A, 1, 90, 0)]
        post(srv, [row(PDF_A, 1, 0)]).close()
        assert get_json(srv, "/api/batch?n=24")["pages"] == []
        assert srv.store.labels[(PDF_A, 1)]["rotation"] == 0
    finally:
        srv.shutdown()
        srv.server_close()
        t.join()


def test_recheck_requires_predictions(tmp_path):
    make_data(tmp_path)
    with pytest.raises(ValueError):
        label.make_server(tmp_path / "pages", None, tmp_path / "labels.csv", tmp_path / "thumbs",
                          port=0, recheck=True)
