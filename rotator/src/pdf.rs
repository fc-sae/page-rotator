//! Lectura y escritura de `/Rotate` con lopdf.
//!
//! Los cambios se guardan como actualización incremental: los bytes originales quedan intactos y
//! solo se agregan al final los diccionarios de página modificados.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use lopdf::{Dictionary, Document, IncrementalDocument, Object, ObjectId};

/// Normaliza un ángulo a 0/90/180/270 (los valores fuera de múltiplos de 90 se redondean).
pub fn normalize(deg: i64) -> u32 {
    let d = deg.rem_euclid(360);
    (((d + 45) / 90 * 90) % 360) as u32
}

fn rotate_of(doc: &Document, dict: &Dictionary) -> Option<i64> {
    match dict.get(b"Rotate").ok()? {
        Object::Integer(i) => Some(*i),
        Object::Real(r) => Some(r.round() as i64),
        Object::Reference(id) => match doc.get_object(*id).ok()? {
            Object::Integer(i) => Some(*i),
            Object::Real(r) => Some(r.round() as i64),
            _ => None,
        },
        _ => None,
    }
}

/// `/Rotate` efectivo de la página: el suyo o el heredado del árbol de páginas.
pub fn page_rotation(doc: &Document, page: ObjectId) -> u32 {
    let mut id = page;
    for _ in 0..64 {
        let Ok(dict) = doc.get_dictionary(id) else { return 0 };
        if let Some(r) = rotate_of(doc, dict) {
            return normalize(r);
        }
        match dict.get(b"Parent") {
            Ok(Object::Reference(parent)) => id = *parent,
            _ => return 0,
        }
    }
    0
}

/// PDF abierto para leer y, si hace falta, ajustar `/Rotate`.
pub struct Pdf {
    inc: IncrementalDocument,
    pages: BTreeMap<u32, ObjectId>,
}

impl Pdf {
    pub fn load(bytes: Vec<u8>) -> Result<Self> {
        let inc = IncrementalDocument::load_from(bytes.as_slice()).context("no se pudo leer el PDF")?;
        let doc = inc.get_prev_documents();
        if doc.is_encrypted() || doc.was_encrypted() || doc.trailer.get(b"Encrypt").is_ok() {
            bail!("el PDF está cifrado; descífralo primero (qpdf --decrypt)");
        }
        let pages = doc.get_pages();
        Ok(Self { inc, pages })
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// `/Rotate` efectivo de cada página, en orden (página 1 primero).
    pub fn rotations(&self) -> Vec<u32> {
        let doc = self.inc.get_prev_documents();
        self.pages.values().map(|&id| page_rotation(doc, id)).collect()
    }

    /// Fija `/Rotate` de la página `index` (0-based) en su propio diccionario.
    pub fn set_rotation(&mut self, index: usize, rotate: u32) -> Result<()> {
        let id = *self.pages.values().nth(index).context("página fuera de rango")?;
        self.inc.opt_clone_object_to_new_document(id)?;
        let obj = self.inc.new_document.get_object_mut(id)?;
        obj.as_dict_mut()?.set("Rotate", Object::Integer(rotate as i64));
        Ok(())
    }

    /// Bytes originales más la actualización incremental (si hubo cambios).
    pub fn to_bytes(mut self) -> Result<Vec<u8>> {
        if self.inc.new_document.objects.is_empty() {
            return Ok(self.inc.get_prev_documents_bytes().to_vec());
        }
        let mut out = Vec::new();
        self.inc.save_to(&mut out).context("no se pudo escribir el PDF")?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_angles() {
        assert_eq!(normalize(0), 0);
        assert_eq!(normalize(-90), 270);
        assert_eq!(normalize(450), 90);
        assert_eq!(normalize(91), 90);
        assert_eq!(normalize(359), 0);
    }
}
