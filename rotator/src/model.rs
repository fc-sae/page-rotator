//! Clasificador ONNX y regla de decisión (misma que `tools/orient.py::decide`).

use anyhow::{Context, Result, anyhow, bail};
use std::sync::OnceLock;

use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

/// Correcciones posibles: grados en sentido horario a aplicar a la página tal como se ve.
pub const CLASSES: [u32; 4] = [0, 90, 180, 270];

#[cfg(embedded_model)]
pub const EMBEDDED: Option<&[u8]> = Some(include_bytes!(concat!(env!("OUT_DIR"), "/model.onnx")));
#[cfg(not(embedded_model))]
pub const EMBEDDED: Option<&[u8]> = None;

/// ONNX Runtime admite un solo entorno por proceso.
fn environment() -> Result<&'static Environment> {
    static ENV: OnceLock<Environment> = OnceLock::new();
    if let Some(env) = ENV.get() {
        return Ok(env);
    }
    let env = ort::init().with_name("page-rotator").build().map_err(|e| anyhow!("ONNX Runtime: {e}"))?;
    Ok(ENV.get_or_init(|| env))
}

pub struct Model {
    session: Session,
    pub size: u32,
}

impl Model {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let err = |e: &dyn std::fmt::Display| anyhow!("no se pudo cargar el modelo ONNX: {e}");
        let session: Session = Session::builder(environment()?)
            .map_err(|e| err(&e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| err(&e))?
            .with_intra_threads(1)
            .map_err(|e| err(&e))?
            .with_inter_threads(1)
            .map_err(|e| err(&e))?
            .commit_from_memory(bytes)
            .map_err(|e| err(&e))?;
        let input = session.inputs().first().context("el modelo no tiene entradas")?;
        let shape = input.dtype().tensor_shape().context("la entrada del modelo no es un tensor")?;
        if shape.len() != 4 || shape[1] != 1 || shape[2] <= 0 || shape[2] != shape[3] {
            bail!("entrada del modelo inesperada {shape:?}; se espera [N, 1, S, S]");
        }
        let size = shape[2] as u32;
        Ok(Self { session, size })
    }

    /// Probabilidades de las 4 clases para cada imagen (`size`×`size`, valores 0–1).
    pub fn predict(&mut self, images: &[Vec<f32>]) -> Result<Vec<[f32; 4]>> {
        let s = self.size as usize;
        let mut data = Vec::with_capacity(images.len() * s * s);
        for img in images {
            data.extend_from_slice(img);
        }
        let input = Tensor::from_array(([images.len(), 1, s, s], data))?;
        let outputs = self.session.run(ort::inputs![input])?;
        let (shape, logits) = outputs[0].try_extract_tensor::<f32>()?;
        if shape.len() != 2 || shape[1] != 4 {
            bail!("salida del modelo inesperada {shape:?}; se espera [N, 4]");
        }
        Ok(logits.chunks(4).map(softmax).collect())
    }
}

pub fn softmax(z: &[f32]) -> [f32; 4] {
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|v| (v - m).exp()).collect();
    let sum: f32 = e.iter().sum();
    [e[0] / sum, e[1] / sum, e[2] / sum, e[3] / sum]
}

fn argmax(p: &[f32; 4]) -> usize {
    (0..4).fold(0, |best, i| if p[i] > p[best] { i } else { best })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision {
    /// Corrección a aplicar (0 = no tocar).
    pub correction: u32,
    /// Corrección más probable según la vista directa, aunque no se aplique.
    pub predicted: u32,
    pub confidence: f32,
    pub consistent: bool,
}

/// `p` sobre la imagen y `p180` sobre la imagen girada 180°: se corrige solo si ambas vistas
/// coinciden y la confianza media alcanza el umbral.
pub fn decide(p: &[f32; 4], p180: &[f32; 4], threshold: f32) -> Decision {
    let (a, b) = (argmax(p), argmax(p180));
    let consistent = b == (a + 2) % 4;
    let confidence = if consistent { (p[a] + p180[b]) / 2.0 } else { p[a].min(p180[b]) };
    let correction = if consistent && confidence >= threshold { CLASSES[a] } else { 0 };
    Decision { correction, predicted: CLASSES[a], confidence, consistent }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decide_matches_python_rule() {
        let p = [0.02, 0.95, 0.02, 0.01];
        let q = [0.01, 0.01, 0.02, 0.96];
        let d = decide(&p, &q, 0.9);
        assert_eq!(d.correction, 90);
        assert!((d.confidence - 0.955).abs() < 1e-6 && d.consistent);
        assert_eq!(decide(&p, &q, 0.99).correction, 0);
        let bad = [0.01, 0.97, 0.01, 0.01];
        let d = decide(&p, &bad, 0.5);
        assert_eq!((d.correction, d.consistent), (0, false));
    }

    #[test]
    fn softmax_sums_to_one() {
        let s = softmax(&[1.0, 2.0, 3.0, 4.0]);
        assert!((s.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(s[3] > s[2]);
    }
}
