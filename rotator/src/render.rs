//! Rasterizado con pdfium al formato de entrada del modelo.
//!
//! Igual que `tools/orient.py`: página tal como se ve (con su `/Rotate`), gris, lado largo a `size`
//! y centrada en un lienzo blanco con offset `(size − lado) / 2`; valores 0–1 (1 = blanco).

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use pdfium_render::prelude::*;

/// Carga libpdfium: `PDFIUM_LIB`, junto al ejecutable o del sistema.
pub fn load_pdfium(explicit: Option<&Path>) -> Result<Pdfium> {
    let mut tried: Vec<PathBuf> = Vec::new();
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(p) = explicit {
        candidates.push(p.to_path_buf());
    }
    if let Some(p) = std::env::var_os("PDFIUM_LIB") {
        candidates.push(PathBuf::from(p));
    }
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
        candidates.push(Pdfium::pdfium_platform_library_name_at_path(&dir));
    }
    for c in candidates {
        if c.is_file() {
            return Pdfium::bind_to_library(&c)
                .map(Pdfium::new)
                .map_err(|e| anyhow!("no se pudo cargar {}: {e:?}", c.display()));
        }
        tried.push(c);
    }
    Pdfium::bind_to_system_library().map(Pdfium::new).map_err(|e| {
        anyhow!(
            "no se encontró libpdfium.so (probé {:?} y el sistema: {e:?}); ponla junto al binario o usa PDFIUM_LIB",
            tried
        )
    })
}

/// Tamaño en píxeles con el lado largo a `size`, conservando la proporción (como `round` de Python).
pub fn fit(w: f32, h: f32, size: u32) -> (u32, u32) {
    let scale = size as f32 / w.max(h).max(1e-3);
    let px = |v: f32| ((v * scale).round() as u32).clamp(1, size);
    (px(w), px(h))
}

/// Coloca una imagen gris (`w`×`h`, bytes) centrada en un lienzo blanco `size`×`size`, valores 0–1.
pub fn letterbox(gray: &[u8], w: u32, h: u32, size: u32) -> Vec<f32> {
    let s = size as usize;
    let mut out = vec![1.0f32; s * s];
    let (ox, oy) = (((size - w) / 2) as usize, ((size - h) / 2) as usize);
    for y in 0..h as usize {
        let row = &gray[y * w as usize..(y + 1) * w as usize];
        let dst = &mut out[(oy + y) * s + ox..(oy + y) * s + ox + w as usize];
        for (d, &v) in dst.iter_mut().zip(row) {
            *d = v as f32 / 255.0;
        }
    }
    out
}

/// Gira 180° un tensor cuadrado (invierte filas y columnas).
pub fn rot180(x: &[f32]) -> Vec<f32> {
    x.iter().rev().copied().collect()
}

/// Rasteriza la página `index` (0-based) a un tensor `size`×`size`.
pub fn page_tensor(doc: &PdfDocument, index: usize, size: u32) -> Result<Vec<f32>> {
    let page = doc
        .pages()
        .get(index as PdfPageIndex)
        .map_err(|e| anyhow!("pdfium no pudo abrir la página {}: {e:?}", index + 1))?;
    let (w, h) = fit(page.width().value, page.height().value, size);
    let cfg = PdfRenderConfig::new()
        .set_target_size(w as Pixels, h as Pixels)
        .use_grayscale_rendering(true)
        .render_form_data(true);
    let bitmap = page
        .render_with_config(&cfg)
        .map_err(|e| anyhow!("pdfium no pudo rasterizar la página {}: {e:?}", index + 1))?;
    let img = bitmap
        .as_image()
        .map_err(|e| anyhow!("imagen inválida en la página {}: {e:?}", index + 1))?
        .to_luma8();
    let (iw, ih) = img.dimensions();
    let (iw, ih) = (iw.min(size), ih.min(size));
    let gray: Vec<u8> = if (iw, ih) == img.dimensions() {
        img.into_raw()
    } else {
        image::imageops::crop_imm(&img, 0, 0, iw, ih).to_image().into_raw()
    };
    Ok(letterbox(&gray, iw, ih, size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_keeps_aspect() {
        assert_eq!(fit(612.0, 792.0, 384), (297, 384));
        assert_eq!(fit(792.0, 612.0, 384), (384, 297));
        assert_eq!(fit(10000.0, 1.0, 384), (384, 1));
    }

    #[test]
    fn letterbox_centers_on_white() {
        let x = letterbox(&[0u8; 2 * 4], 2, 4, 4);
        assert_eq!(x.len(), 16);
        assert_eq!(x[0], 1.0); // relleno
        assert_eq!(x[1], 0.0); // contenido en columnas 1–2
        assert_eq!(x[3], 1.0);
    }

    #[test]
    fn rot180_reverses() {
        assert_eq!(rot180(&[1.0, 2.0, 3.0, 4.0]), vec![4.0, 3.0, 2.0, 1.0]);
    }
}
