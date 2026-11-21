//! Depuración: guarda como PGM el tensor de entrada de cada página. `dump IN.pdf OUT_DIR [size]`
use std::io::Write;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let size: u32 = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(384);
    let pdfium = page_rotator::render::load_pdfium(None)?;
    let bytes = std::fs::read(&a[1])?;
    let doc = pdfium.load_pdf_from_byte_slice(&bytes, None).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    for i in 0..doc.pages().len() as usize {
        let x = page_rotator::render::page_tensor(&doc, i, size)?;
        let mut f = std::fs::File::create(format!("{}/{:03}.pgm", a[2], i + 1))?;
        write!(f, "P5\n{size} {size}\n255\n")?;
        f.write_all(&x.iter().map(|v| (v * 255.0).round() as u8).collect::<Vec<u8>>())?;
    }
    Ok(())
}
