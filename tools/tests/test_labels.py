import csv

import pytest
from PIL import Image

import labels


def _manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pdf_id", "page", "status", "rotate", "width_pt", "height_pt", "width_px", "height_px", "path", "error"])
        for pdf_id, page, status in rows:
            w.writerow([pdf_id, page, status, 0, 1, 1, 1, 1, "", "" if status == "ok" else "boom"])


def _clusters(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pdf_id", "page", "cluster", "dist"])
        for pdf_id, page, cluster in rows:
            w.writerow([pdf_id, page, cluster, 0.1])


def test_add_and_reload(tmp_path):
    p = tmp_path / "labels.csv"
    s = labels.LabelStore(p)
    s.add([
        {"pdf_id": "a", "page": 1, "rotation": 90, "status": "ok"},
        {"pdf_id": "b", "page": 2, "rotation": 0, "status": "skip"},
    ])
    s2 = labels.LabelStore(p)
    assert len(s2) == 2
    assert ("a", 1) in s2
    assert s2.labels[("a", 1)]["rotation"] == 90
    assert isinstance(s2.labels[("b", 2)]["page"], int)
    assert s2.labels[("b", 2)]["status"] == "skip"


def test_last_row_wins(tmp_path):
    p = tmp_path / "labels.csv"
    s = labels.LabelStore(p)
    s.add([{"pdf_id": "a", "page": 1, "rotation": 90, "status": "ok"}])
    s.add([{"pdf_id": "a", "page": 1, "rotation": 0, "status": "ok"}])
    assert s.labels[("a", 1)]["rotation"] == 0
    assert len(s) == 1
    assert len(p.read_text(encoding="utf-8").strip().splitlines()) == 3
    assert labels.LabelStore(p).labels[("a", 1)]["rotation"] == 0


def test_add_rejects_invalid(tmp_path):
    p = tmp_path / "labels.csv"
    s = labels.LabelStore(p)
    s.add([{"pdf_id": "a", "page": 1, "rotation": 90, "status": "ok"}])
    before = p.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        s.add([{"pdf_id": "a", "page": 2, "rotation": 45, "status": "ok"}])
    with pytest.raises(ValueError):
        s.add([{"pdf_id": "a", "page": 2, "rotation": 0, "status": "x"}])
    with pytest.raises(ValueError):
        s.add([
            {"pdf_id": "a", "page": 3, "rotation": 0, "status": "ok"},
            {"pdf_id": "a", "page": 4, "rotation": 45, "status": "ok"},
        ])
    assert p.read_text(encoding="utf-8") == before
    assert len(s) == 1


def test_round_robin_by_cluster(tmp_path):
    m, c = tmp_path / "m.csv", tmp_path / "c.csv"
    _manifest(m, [("a", 1, "ok"), ("a", 2, "ok"), ("a", 3, "ok"), ("b", 1, "ok"), ("c", 1, "ok")])
    _clusters(c, [("a", 1, 0), ("a", 2, 0), ("a", 3, 0), ("b", 1, 1)])
    q = labels.build_queue(m, c, seed=1)
    assert len(q) == 5
    groups = {("a", 1): 0, ("a", 2): 0, ("a", 3): 0, ("b", 1): 1, ("c", 1): "x"}
    assert len({groups[k] for k in q[:3]}) == 3
    assert q == labels.build_queue(m, c, seed=1)


def test_queue_without_clusters_file(tmp_path):
    m = tmp_path / "m.csv"
    _manifest(m, [("a", 1, "ok"), ("a", 2, "ok"), ("b", "", "error")])
    q = labels.build_queue(m, None)
    assert sorted(q) == [("a", 1), ("a", 2)]


PDF = "1QVHKvpf9r_QaydcdjZpTFRhJlrVklTfh"


def _png(pages_dir, pdf_id=PDF, page=1, size=(1275, 1650)):
    d = pages_dir / pdf_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{page:03d}.png"
    Image.new("L", size, 255).save(p)
    return p


def test_thumb_created_once(tmp_path):
    pages, thumbs = tmp_path / "pages", tmp_path / "thumbs"
    _png(pages)
    t = labels.image_path(pages, thumbs, PDF, 1, "thumb")
    assert t == thumbs / PDF / "001.jpg"
    with Image.open(t) as im:
        assert im.size == (247, 320)
        assert im.format == "JPEG"
    mtime = t.stat().st_mtime_ns
    assert labels.image_path(pages, thumbs, PDF, 1, "thumb") == t
    assert t.stat().st_mtime_ns == mtime
    assert not list((thumbs / PDF).glob("*.part"))


def test_full_returns_png(tmp_path):
    pages, thumbs = tmp_path / "pages", tmp_path / "thumbs"
    png = _png(pages)
    assert labels.image_path(pages, thumbs, PDF, 1, "full") == png


def test_image_rejects_bad_ids(tmp_path):
    pages, thumbs = tmp_path / "pages", tmp_path / "thumbs"
    _png(pages)
    assert labels.image_path(pages, thumbs, "../../etc", 1, "full") is None
    assert labels.image_path(pages, thumbs, "abc", 1, "full") is None
    assert labels.image_path(pages, thumbs, PDF, 0, "full") is None
    assert labels.image_path(pages, thumbs, PDF, 1, "big") is None
    assert labels.image_path(pages, thumbs, PDF, 2, "full") is None
    assert labels.image_path(pages, thumbs, PDF, 2, "thumb") is None
