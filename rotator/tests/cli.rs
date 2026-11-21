//! Pruebas de la CLI con PDFs sintéticos y un modelo falso (tests/fixtures/edge.onnx) que detecta
//! la barra negra de la parte superior de la página.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use lopdf::content::{Content, Operation};
use lopdf::{Document, Object, Stream, dictionary};
use page_rotator::pdf::page_rotation;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn pdfium_lib() -> PathBuf {
    if let Some(p) = std::env::var_os("PDFIUM_LIB") {
        return PathBuf::from(p);
    }
    let venv = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/.venv/lib");
    for entry in std::fs::read_dir(&venv).expect("falta tools/.venv (uv sync) o PDFIUM_LIB") {
        let p = entry.unwrap().path().join("site-packages/pypdfium2_raw/libpdfium.so");
        if p.exists() {
            return p;
        }
    }
    panic!("no encontré libpdfium.so; define PDFIUM_LIB");
}

struct PageSpec {
    rotate: Option<i64>,
    blank: bool,
    landscape: bool,
}

fn page(rotate: Option<i64>) -> PageSpec {
    PageSpec { rotate, blank: false, landscape: false }
}

/// PDF con una barra negra arriba y un bloque a la izquierda en cada página (contenido derecho).
fn make_pdf(pages: &[PageSpec], parent_rotate: Option<i64>) -> Vec<u8> {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let mut kids = vec![];
    for spec in pages {
        let (w, h) = if spec.landscape { (792, 612) } else { (612, 792) };
        let ops = if spec.blank {
            vec![]
        } else {
            vec![
                Operation::new("g", vec![0.into()]),
                Operation::new("re", vec![0.into(), (h - 60).into(), w.into(), 60.into()]),
                Operation::new("f", vec![]),
                Operation::new("re", vec![0.into(), (h / 2).into(), 60.into(), (h / 2 - 60).into()]),
                Operation::new("f", vec![]),
            ]
        };
        let content = doc.add_object(Stream::new(dictionary! {}, Content { operations: ops }.encode().unwrap()));
        let mut dict = dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "MediaBox" => vec![0.into(), 0.into(), w.into(), h.into()],
            "Contents" => content,
        };
        if let Some(r) = spec.rotate {
            dict.set("Rotate", r);
        }
        kids.push(Object::Reference(doc.add_object(dict)));
    }
    let mut pages_dict = dictionary! { "Type" => "Pages", "Count" => kids.len() as i64, "Kids" => kids };
    if let Some(r) = parent_rotate {
        pages_dict.set("Rotate", r);
    }
    doc.objects.insert(pages_id, Object::Dictionary(pages_dict));
    let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
    doc.trailer.set("Root", catalog);
    let mut out = Vec::new();
    doc.save_to(&mut out).unwrap();
    out
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_page-rotator"))
        .env("PDFIUM_LIB", pdfium_lib())
        .arg("--model")
        .arg(fixture("edge.onnx"))
        .args(args)
        .output()
        .unwrap()
}

fn rotations(path: &Path) -> Vec<u32> {
    let doc = Document::load(path).unwrap();
    doc.get_pages().values().map(|&id| page_rotation(&doc, id)).collect()
}

fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, bytes).unwrap();
    p
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[test]
fn corrects_single_rotated_page() {
    let dir = tempfile::tempdir().unwrap();
    let input = write(dir.path(), "in.pdf", &make_pdf(&[page(Some(90))], None));
    let out = dir.path().join("out.pdf");
    let o = run(&["--json", s(&input), s(&out)]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(rotations(&out), vec![0]);
    let json = String::from_utf8(o.stdout).unwrap();
    assert!(json.contains("\"rotate\":90") && json.contains("\"correction\":270") && json.contains("\"changed\":1"), "{json}");
    // los bytes originales se conservan: actualización incremental
    let orig = std::fs::read(&input).unwrap();
    assert!(std::fs::read(&out).unwrap().starts_with(&orig));
}

#[test]
fn mixed_multipage_all_end_upright() {
    let dir = tempfile::tempdir().unwrap();
    let pages: Vec<PageSpec> = [None, Some(90), Some(180), Some(270), Some(-90)].into_iter().map(page).collect();
    let input = write(dir.path(), "in.pdf", &make_pdf(&pages, None));
    let out = dir.path().join("out.pdf");
    let o = run(&["-q", s(&input), s(&out)]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(o.stderr.is_empty());
    assert_eq!(rotations(&out), vec![0, 0, 0, 0, 0]);
}

#[test]
fn landscape_page_is_corrected() {
    let dir = tempfile::tempdir().unwrap();
    let spec = PageSpec { rotate: Some(270), blank: false, landscape: true };
    let input = write(dir.path(), "in.pdf", &make_pdf(&[spec], None));
    let out = dir.path().join("out.pdf");
    assert!(run(&["-q", s(&input), s(&out)]).status.success());
    assert_eq!(rotations(&out), vec![0]);
}

#[test]
fn inherited_rotate_is_corrected_on_the_page() {
    let dir = tempfile::tempdir().unwrap();
    let input = write(dir.path(), "in.pdf", &make_pdf(&[page(None), page(Some(0))], Some(180)));
    assert_eq!(rotations(&input), vec![180, 0]);
    let out = dir.path().join("out.pdf");
    assert!(run(&["-q", s(&input), s(&out)]).status.success());
    assert_eq!(rotations(&out), vec![0, 0]);
}

#[test]
fn blank_and_upright_pages_untouched_and_bytes_identical() {
    let dir = tempfile::tempdir().unwrap();
    let blank = PageSpec { rotate: Some(90), blank: true, landscape: false };
    let bytes = make_pdf(&[blank, page(None)], None);
    let input = write(dir.path(), "in.pdf", &bytes);
    let out = dir.path().join("out.pdf");
    let o = run(&["--json", s(&input), s(&out)]);
    assert!(o.status.success());
    assert_eq!(std::fs::read(&out).unwrap(), bytes);
    assert!(String::from_utf8(o.stdout).unwrap().contains("\"changed\":0"));
}

#[test]
fn threshold_one_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let input = write(dir.path(), "in.pdf", &make_pdf(&[page(Some(90))], None));
    let out = dir.path().join("out.pdf");
    assert!(run(&["-q", "--threshold", "1", s(&input), s(&out)]).status.success());
    assert_eq!(rotations(&out), vec![90]);
}

#[test]
fn in_place_and_dry_run() {
    let dir = tempfile::tempdir().unwrap();
    let input = write(dir.path(), "in.pdf", &make_pdf(&[page(Some(90))], None));
    let o = run(&["-q", "--dry-run", s(&input)]);
    assert!(o.status.success());
    assert_eq!(rotations(&input), vec![90]);
    assert!(run(&["-q", s(&input), s(&input)]).status.success());
    assert_eq!(rotations(&input), vec![0]);
    let leftovers: Vec<_> = std::fs::read_dir(dir.path()).unwrap().collect();
    assert_eq!(leftovers.len(), 1, "no deben quedar temporales");
}

#[test]
fn output_passes_qpdf_check() {
    if Command::new("qpdf").arg("--version").output().is_err() {
        eprintln!("qpdf no instalado; se omite");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let input = write(dir.path(), "in.pdf", &make_pdf(&[page(Some(90)), page(None)], None));
    let out = dir.path().join("out.pdf");
    assert!(run(&["-q", s(&input), s(&out)]).status.success());
    let check = Command::new("qpdf").arg("--check").arg(&out).output().unwrap();
    assert!(check.status.success(), "{}", String::from_utf8_lossy(&check.stdout));
}

#[test]
fn encrypted_pdf_fails_with_code_1() {
    let dir = tempfile::tempdir().unwrap();
    let plain = write(dir.path(), "plain.pdf", &make_pdf(&[page(Some(90))], None));
    let enc = dir.path().join("enc.pdf");
    match Command::new("qpdf").args(["--encrypt", "", "owner", "256", "--"]).arg(&plain).arg(&enc).status() {
        Ok(st) if st.success() => {}
        _ => {
            eprintln!("qpdf no disponible; se omite");
            return;
        }
    }
    let out = dir.path().join("out.pdf");
    let o = run(&[s(&enc), s(&out)]);
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).contains("cifrado"));
    assert!(!out.exists());
}

#[test]
fn bad_inputs_exit_codes() {
    let dir = tempfile::tempdir().unwrap();
    let empty = write(dir.path(), "empty.pdf", b"");
    let junk = write(dir.path(), "junk.pdf", b"esto no es un PDF");
    let out = dir.path().join("out.pdf");
    assert_eq!(run(&[s(&empty), s(&out)]).status.code(), Some(1));
    assert_eq!(run(&[s(&junk), s(&out)]).status.code(), Some(1));
    assert_eq!(run(&[s(&dir.path().join("no.pdf")), s(&out)]).status.code(), Some(1));
    assert!(!out.exists());
    let ok = write(dir.path(), "ok.pdf", &make_pdf(&[page(None)], None));
    assert_eq!(run(&["--threshold", "2", s(&ok), s(&out)]).status.code(), Some(2));
    assert_eq!(run(&[s(&ok)]).status.code(), Some(2));
    assert_eq!(run(&["--nope", s(&ok), s(&out)]).status.code(), Some(2));
}

#[test]
fn bad_model_exit_code_1() {
    let dir = tempfile::tempdir().unwrap();
    let ok = write(dir.path(), "ok.pdf", &make_pdf(&[page(None)], None));
    let bad = write(dir.path(), "bad.onnx", b"nope");
    let o = Command::new(env!("CARGO_BIN_EXE_page-rotator"))
        .env("PDFIUM_LIB", pdfium_lib())
        .args(["--model", s(&bad), s(&ok), s(&dir.path().join("o.pdf"))])
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(1));
}

/// Completa la línea `words` (la última palabra es la que se está escribiendo) con el script de bash.
fn bash_complete(words: &[&str]) -> Vec<String> {
    let script = Command::new(env!("CARGO_BIN_EXE_page-rotator")).args(["--completions", "bash"]).output().unwrap();
    assert!(script.status.success());
    let dir = tempfile::tempdir().unwrap();
    let path = write(dir.path(), "comp.bash", &script.stdout);
    let line = words.join(" ");
    let cmd = format!(
        "source {}; COMP_WORDS=({line}); COMP_CWORD={}; COMP_LINE='{line}'; COMP_POINT=${{#COMP_LINE}}; \
         _page__rotator page-rotator '{}' '{}'; printf '%s\\n' \"${{COMPREPLY[@]}}\"",
        s(&path),
        words.len() - 1,
        words[words.len() - 1],
        words[words.len() - 2],
    );
    let o = Command::new("bash").arg("-c").arg(cmd).current_dir(dir.path()).output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout).unwrap().lines().map(str::to_string).collect()
}

#[test]
fn bash_completion_suggests_options() {
    let got = bash_complete(&["page-rotator", "--th"]);
    assert_eq!(got, vec!["--threshold"]);
    let all = bash_complete(&["page-rotator", "--"]);
    for opt in ["--dry-run", "--json", "--model", "--pdfium", "--quiet", "--completions"] {
        assert!(all.iter().any(|o| o == opt), "falta {opt} en {all:?}");
    }
    assert_eq!(bash_complete(&["page-rotator", "--completions", "ba"]), vec!["bash"]);
}

#[test]
fn completions_do_not_require_input_but_normal_runs_do() {
    let o = Command::new(env!("CARGO_BIN_EXE_page-rotator")).args(["--completions", "bash"]).output().unwrap();
    assert!(o.status.success());
    assert!(String::from_utf8_lossy(&o.stdout).contains("complete -F _page__rotator"));
    let o = Command::new(env!("CARGO_BIN_EXE_page-rotator")).output().unwrap();
    assert_eq!(o.status.code(), Some(2));
}
