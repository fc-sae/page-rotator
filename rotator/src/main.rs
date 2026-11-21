use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use clap::Parser;

use page_rotator::{PageResult, model, process, render};

/// Detecta la orientación de cada página de un PDF y corrige /Rotate (sin re-rasterizar).
///
/// Códigos de salida: 0 OK, 1 error de PDF o de modelo, 2 argumentos inválidos.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// No imprimir el resumen.
    #[arg(short, long)]
    quiet: bool,
    /// Confianza mínima para corregir una página (0–1).
    #[arg(long, default_value_t = 0.9)]
    threshold: f32,
    /// Modelo ONNX (por defecto, el incrustado en el binario).
    #[arg(long)]
    model: Option<PathBuf>,
    /// libpdfium.so (por defecto: PDFIUM_LIB, junto al binario o del sistema).
    #[arg(long)]
    pdfium: Option<PathBuf>,
    /// Solo analizar; no escribir OUT.
    #[arg(long)]
    dry_run: bool,
    /// Reporte por página en JSON (stdout).
    #[arg(long)]
    json: bool,
    /// PDF de entrada.
    input: PathBuf,
    /// PDF de salida (puede ser el mismo que la entrada). Opcional con --dry-run.
    output: Option<PathBuf>,
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn report_json(input: &Path, results: &[PageResult]) -> String {
    let pages: Vec<String> = results
        .iter()
        .map(|r| {
            format!(
                "{{\"page\":{},\"rotate\":{},\"predicted\":{},\"confidence\":{:.4},\"consistent\":{},\"correction\":{},\"new_rotate\":{}}}",
                r.page, r.rotate, r.decision.predicted, r.decision.confidence, r.decision.consistent,
                r.decision.correction, r.new_rotate
            )
        })
        .collect();
    let changed = results.iter().filter(|r| r.new_rotate != r.rotate).count();
    format!(
        "{{\"input\":\"{}\",\"pages\":[{}],\"changed\":{}}}",
        json_escape(&input.to_string_lossy()),
        pages.join(","),
        changed
    )
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let name = path.file_name().context("ruta de salida inválida")?.to_string_lossy();
    let tmp = dir.join(format!(".{name}.{}.tmp", std::process::id()));
    std::fs::write(&tmp, bytes).with_context(|| format!("no se pudo escribir {}", tmp.display()))?;
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })?;
    Ok(())
}

fn run(args: &Args) -> Result<()> {
    let model_bytes = match &args.model {
        Some(p) => std::fs::read(p).with_context(|| format!("no se pudo leer el modelo {}", p.display()))?,
        None => match model::EMBEDDED {
            Some(b) => b.to_vec(),
            None => bail!("este binario no trae modelo incrustado; usa --model"),
        },
    };
    let mut model = model::Model::from_bytes(&model_bytes)?;
    let pdfium = render::load_pdfium(args.pdfium.as_deref())?;
    let bytes = std::fs::read(&args.input).with_context(|| format!("no se pudo leer {}", args.input.display()))?;
    let (doc, results) = process(&bytes, &pdfium, &mut model, args.threshold)
        .with_context(|| format!("{}", args.input.display()))?;
    if args.json {
        println!("{}", report_json(&args.input, &results));
    }
    let changed: Vec<&PageResult> = results.iter().filter(|r| r.new_rotate != r.rotate).collect();
    if !args.quiet {
        let detail: Vec<String> = changed.iter().map(|r| format!("p{} {}°→{}°", r.page, r.rotate, r.new_rotate)).collect();
        eprintln!(
            "{}: {} páginas, {} corregidas{}{}",
            args.input.display(),
            results.len(),
            changed.len(),
            if detail.is_empty() { "" } else { " (" },
            if detail.is_empty() { String::new() } else { detail.join(", ") + ")" }
        );
    }
    if let Some(out) = &args.output {
        if !args.dry_run {
            write_atomic(out, &doc.to_bytes()?)?;
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse();
    if !(0.0..=1.0).contains(&args.threshold) {
        eprintln!("error: --threshold debe estar entre 0 y 1");
        return ExitCode::from(2);
    }
    if args.output.is_none() && !args.dry_run {
        eprintln!("error: falta OUT.pdf (o usa --dry-run)");
        return ExitCode::from(2);
    }
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("page-rotator: {e:#}");
            ExitCode::from(1)
        }
    }
}
