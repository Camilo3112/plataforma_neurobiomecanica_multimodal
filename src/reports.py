"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       GENERACIÓN DE REPORTES Y TABLAS                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/reports.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.


-----------------------------------------
Genera salidas tabulares y reportes reproducibles. Las métricas se conservan
con
unidad, paciente, etapa y fuente. El modelo de reporte prioriza trazabilidad:
cada resultado numérico deriva de una tabla intermedia y puede auditarse por
ruta
y nombre de variable.

Trazabilidad de resultados
--------------------------
Los datos crudos permanecen separados de los derivados. Las salidas se escriben
con nombre de paciente, etapa, módulo y espacio de referencia para facilitar
revisión, comparación longitudinal y reproducción de la corrida.
"""

###############################################################################################################################
###############################################################################################################################

# ══════════════════════════════════════════════════════════════════════════════
#  0. IMPORTACIONES, CONFIGURACIÓN Y FUNCIONES DEL MÓDULO
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd

from .config import PipelineConfig
from .io_utils import safe_mkdir, save_dataframe


def _write_excel_report(out_path: Path, available: dict[str, Path]) -> Path:
    """Escribe Excel global. Si el archivo esta abierto en Excel/OneDrive, usa copia con timestamp."""
    safe_mkdir(out_path.parent)

    def _write(path: Path) -> None:
        with pd.ExcelWriter(path) as writer:
            for name, src_path in available.items():
                try:
                    df = pd.read_csv(src_path)
                    df.to_excel(writer, sheet_name=name[:31], index=False)
                except Exception as exc:
                    pd.DataFrame([{"error": str(exc), "path": str(src_path)}]).to_excel(
                        writer, sheet_name=f"error_{name}"[:31], index=False
                    )

    try:
        _write(out_path)
        return out_path
    except PermissionError:
        alt = out_path.with_name(f"{out_path.stem}_{datetime.now():%Y%m%d_%H%M%S}.xlsx")
        print("ADVERTENCIA: no se pudo sobrescribir el Excel global.")
        print("Probablemente esta abierto en Excel o bloqueado por Windows/OneDrive:")
        print(f"  {out_path}")
        print("Guardando una copia alternativa en:")
        print(f"  {alt}")
        _write(alt)
        return alt


def build_global_report(cfg: PipelineConfig) -> Path | None:
    """Une todos los CSV globales en un Excel resumen."""
    out_path = cfg.results_root() / "reporte_global_integrado.xlsx"
    csvs = {
        "biomecanica": cfg.results_root() / "resumen_global_biomecanica.csv",
        "tomografia": cfg.results_root() / "resumen_global_tomografia.csv",
        "mapas": cfg.results_root() / "resumen_global_mapas.csv",
        "resonancias": cfg.results_root() / "resumen_global_resonancias.csv",
        "correlaciones": cfg.results_root() / "resumen_global_correlaciones_volumenes.csv",
        "comparacion": cfg.results_root() / "resumen_global_comparacion_antes_vs_despues.csv",
        "estructura_funcion": cfg.results_root() / "resumen_global_estructura_funcion.csv",
        "morfometria_interna": cfg.results_root() / "resumen_global_morfometria_interna.csv",
        "morfometria_externa": cfg.results_root() / "resumen_global_morfometria_externa.csv",
        "consolidado": cfg.results_root() / "resumen_global_consolidado_pacientes.csv",
    }
    available = {name: path for name, path in csvs.items() if path.exists()}
    if not available:
        return None
    return _write_excel_report(out_path, available)


def write_readme_results(cfg: PipelineConfig) -> Path:
    path = cfg.results_root() / "LEEME_RESULTADOS.txt"
    safe_mkdir(path.parent)
    text = f"""
SUITE INTEGRADA VCE · RESULTADOS
=================================

Carpeta raíz del proyecto:
{cfg.project_root}

Estructura generada por paciente y etapa:
resultados/<paciente>/<Antes|Despues>/
  emg/prueba_1..3/
  dinamometria/prueba_1..3/
  tomografia/
  mapas/
  resonancias/
  correlaciones/
  reportes/

Archivos clave:
- biomecanica_metricas_todas_las_pruebas.csv/.xlsx: métricas de las 16 fases por prueba.
- analisis_avanzado/: escalogramas CWT-like, coherencia, entropía multiescala, Hurst, PLV y matrices .npy por fase.
- _estado_pipeline/checkpoints.json: permite reanudar sin repetir lo ya terminado.
- _estado_pipeline/pipeline_eventos.log: historial de START/DONE/SKIP/FAIL por tarea.
- volumenes_y_areas.csv: volúmenes musculares y áreas transversales desde TAC.
- mask_*.nii.gz y masks_musculos_labelmap.nii.gz: volúmenes segmentados para abrir en 3D Slicer/FSL/MRIcron.
- manifest_series_detectadas.csv: inventario automático de series DICOM por etapa.
- resumen_series_por_tipo.csv y resumen_metricas_por_tipo.csv: conteos y métricas agrupadas por tipo.
- *_volumen.nii.gz: conversión de cada serie DICOM/NIfTI procesada.
- *_wavelet_energy.nii.gz: volumen de energía multiescala para mapas/resonancias.
- *_topografia_3d.png: gráfico de superficie tipo topografía.
- *_histograma.png y *_mip.png: previsualización extra de distribución de intensidad y proyección máxima.
- *_fmri_temporal_acf_psd.png: análisis temporal exploratorio para stacks fMRI/motor.
- *_mask_p95.nii.gz y *_mask_p99.nii.gz: máscaras exploratorias de alta señal/activación.
- vce_heuristica_*.nii.gz: ROI exploratoria de vía corticoespinal cuando no se puede usar una serie explícita.
- *_mascara_tracto_corticoespinal_visual.nii.gz: máscara exploratoria generada desde serie TRATO/TRACTO CORTICOESPINAL si existe.
- deficit_*.nii.gz: comparación z-score paciente vs sano cuando hay volúmenes compatibles.
- consolidado/<sujeto>_todos_los_datos_largo.csv: tabla larga con todas las métricas de Antes, Despues, correlaciones y comparación.
- consolidado/<sujeto>_tabla_ancha_metricas.csv: unión heterogénea de filas originales de CSV/JSON de resultados.
- consolidado/<sujeto>_diccionario_metricas.csv: explicación y unidad/escala de cada métrica.
- consolidado/<sujeto>_resumen_estadistico_metricas.csv: resumen numérico por etapa, modalidad y métrica.

Organización especial de resonancias:
resonancias/
  anatomica/t1|t2_tse|t2_tirm_flair/
  dti_difusion/fa|adc|rd|ad|trace_b0|tensor|colfa|dwi/
  fmri_motor/derecha|izquierda/mapa_activacion|bold_raw|moco|design/
  via_corticoespinal/vce_explicita|tractografia/
  resting_state/, swi/, angio_tof/, fieldmap/, postproceso/, scout/, otros/

Reanudación:
- Ejecución normal: salta automáticamente tareas terminadas.
- --force: reprocesa todo desde cero.
- --skip-stuck: salta tareas que quedaron running/failed y continúa con lo demás.

ADVERTENCIA VCE:
Tu manifest muestra series de difusión/DTI y también series visuales llamadas TRACTOGRAFÍA o TRACTO CORTICOESPINAL.
La suite las convierte a NIfTI y genera métricas, pero la identificación clínica definitiva de la vía corticoespinal requiere validar orientación, tractografía cuantitativa y/o registro anatómico en un visor médico.
""".strip()
    path.write_text(text, encoding="utf-8")
    return path
