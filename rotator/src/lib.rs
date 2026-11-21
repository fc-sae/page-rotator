//! page-rotator: detecta la orientación de cada página y corrige `/Rotate` sin re-rasterizar.

pub mod model;
pub mod pdf;
pub mod render;

use anyhow::{Result, anyhow, bail};
use pdfium_render::prelude::Pdfium;

use model::{Decision, Model};
use pdf::Pdf;

#[derive(Debug, Clone)]
pub struct PageResult {
    /// Número de página (1-based).
    pub page: usize,
    /// `/Rotate` original (efectivo, incluye el heredado).
    pub rotate: u32,
    pub decision: Decision,
    /// `/Rotate` final: (original + corrección) mod 360.
    pub new_rotate: u32,
}

/// Analiza todas las páginas y aplica las correcciones al `Pdf` (en memoria).
pub fn process(bytes: &[u8], pdfium: &Pdfium, model: &mut Model, threshold: f32) -> Result<(Pdf, Vec<PageResult>)> {
    let mut doc = Pdf::load(bytes.to_vec())?;
    let rendered = pdfium
        .load_pdf_from_byte_slice(bytes, None)
        .map_err(|e| anyhow!("pdfium no pudo abrir el PDF: {e:?}"))?;
    let n = doc.page_count();
    if rendered.pages().len() as usize != n {
        bail!("número de páginas distinto entre lopdf ({n}) y pdfium ({})", rendered.pages().len());
    }
    let rotations = doc.rotations();
    let mut results = Vec::with_capacity(n);
    for (i, &rotate) in rotations.iter().enumerate() {
        let x = render::page_tensor(&rendered, i, model.size)?;
        let x180 = render::rot180(&x);
        let probs = model.predict(&[x, x180])?;
        let decision = model::decide(&probs[0], &probs[1], threshold);
        let new_rotate = (rotate + decision.correction) % 360;
        if new_rotate != rotate {
            doc.set_rotation(i, new_rotate)?;
        }
        results.push(PageResult { page: i + 1, rotate, decision, new_rotate });
    }
    Ok((doc, results))
}
