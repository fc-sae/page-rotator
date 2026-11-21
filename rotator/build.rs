//! Incrusta el modelo ONNX en el binario si existe.
//!
//! Ruta: variable PAGE_ROTATOR_MODEL o ../data/models/orientation.onnx. Sin modelo, el binario
//! compila igual y exige --model.
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=PAGE_ROTATOR_MODEL");
    println!("cargo:rustc-check-cfg=cfg(embedded_model)");
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let model = std::env::var_os("PAGE_ROTATOR_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("../data/models/orientation.onnx"));
    println!("cargo:rerun-if-changed={}", model.display());
    if model.is_file() {
        let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("model.onnx");
        std::fs::copy(&model, &out).expect("no se pudo copiar el modelo");
        println!("cargo:rustc-cfg=embedded_model");
    } else {
        println!("cargo:warning=sin modelo en {}: el binario exigirá --model", model.display());
    }
}
