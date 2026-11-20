# Herramienta de etiquetado de orientación (etapa 3b) — Diseño

## Para qué
Héctor marca la rotación correcta de ~1000 páginas iniciales, de todos los tipos de constancia, para
entrenar el primer modelo. Las páginas vienen tal como las envían los alumnos (sin orientar).
La mayoría ya está derecha, así que el flujo debe hacer barato confirmar y rápido corregir.

**Éxito:** etiquetar una pantalla de 24 páginas mayormente derechas toma unos segundos (solo Enter);
corregir una página girada es una tecla; nunca se pierde trabajo.

**No-objetivos:** varios usuarios a la vez, acceso desde otra máquina, editar los PDFs.

## Enfoques
1. **Elegido — servidor local con la biblioteca estándar de Python + una página HTML/JS.** Sin
   dependencias nuevas, control total del teclado y de la cuadrícula.
2. Streamlit / Gradio: dependencia nueva, recargas de página completas, difícil manejar una cuadrícula
   con teclado.
3. Notebook con widgets: igual de torpe con teclado y requiere Jupyter.

## Diseño (aprobado)
- `tools/label.py`: `uv run python label.py` levanta un servidor en `127.0.0.1` y abre el navegador.
- **Cola:** páginas `ok` de `data/pages/manifest.csv` aún sin etiqueta, alternando entre clústeres de
  `data/clusters/clusters.csv` (orden aleatorio fijo dentro de cada clúster) para que las primeras
  1000 cubran todos los tipos. Las páginas sin clúster (renderizadas después) van en su propio grupo.
- **Pantalla:** 24 miniaturas. Cada una se muestra *ya con la corrección aplicada*. Todas empiezan en
  "derecha" (0°).
- **Teclas:** ←→↑↓ mover · **R ↷** girar 90° a la derecha · **E ↶** girar 90° a la izquierda ·
  **X** duda / en blanco · **Espacio** ver grande · **Enter** guardar y siguiente pantalla ·
  **Retroceso** volver a la pantalla anterior para corregir. Clic en la miniatura = seleccionar;
  botones ↶ ↷ en cada miniatura para usar el mouse.
- **Ayuda visual del giro:** la barra de ayuda muestra `R ↷ derecha` y `E ↶ izquierda`; cada miniatura
  corregida lleva una insignia con el sentido y los grados: `↷ 90°`, `↻ 180°`, `↶ 90°`, o `? duda`.
- **`data/labels.csv`:** `pdf_id, page, rotation, status, labeled_at`. `rotation` = grados en sentido
  horario que hay que aplicar a la página *tal como se ve* (con su `/Rotate`) para dejarla derecha:
  0/90/180/270, igual que la CLI (`nuevo /Rotate = (actual + rotation) mod 360`). `status` = `ok` o
  `skip` (duda/en blanco, fuera del entrenamiento). Solo se agregan filas; la última fila de cada
  página gana. Se escribe en cada Enter: se puede cerrar y seguir después.
- **Miniaturas:** se generan al vuelo (lado largo 320 px, JPEG) y se guardan en `data/thumbs/`.
- **Seguridad:** solo escucha en `127.0.0.1`; los parámetros de ruta se validan (ID de Drive y número
  de página) para no servir archivos fuera de `data/`.

## A criterio del implementador
Estilos, tamaño exacto de la cuadrícula según la ventana, puerto por defecto, formato del contador
de avance.

## Después (no en esta etapa)
Modos para 3c (cola de páginas dudosas del modelo) y 3d (revisión del test dorado) reutilizarán la
misma interfaz con otra cola.
