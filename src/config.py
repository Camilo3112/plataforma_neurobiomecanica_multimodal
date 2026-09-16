"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   CONFIGURACIÓN COMPUTACIONAL DEL PIPELINE                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/config.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.

Fundamento físico-matemático implementado
-----------------------------------------
Define parámetros del pipeline como un modelo de configuración reproducible.
Las
rutas, umbrales, etapas, sujetos y banderas de ejecución se encapsulan para
que
una corrida pueda repetirse con el mismo vector de parámetros θ. Esto reduce
ambigüedad entre análisis manuales y ejecuciones por lote.

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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class PipelineConfig:
    """Configuración central de la suite.

    Cambia aquí solo si tu estructura de carpetas cambia.
    Por defecto asume:
        D:\\EAFIT\\01-2026\\proyecto\\datos\\paciente 1\\Antes\\EMG\\derecha_1.csv
    """

    project_root: Path
    data_root_override: Optional[Path] = None
    results_root_override: Optional[Path] = None
    data_dirname: str = "datos"
    results_dirname: str = "resultados"
    patients: tuple[str, ...] = ("paciente 1", "paciente 2")
    control_name: str = "sano"
    stages: tuple[str, ...] = ("Antes", "Despues")
    tests: tuple[int, ...] = (1, 2, 3)

    # Reanudación/checkpoint. Si el análisis se cae, al ejecutar otra vez se salta
    # lo ya completado con la misma firma de entrada. Usa --force para repetir todo.
    resume: bool = True
    force: bool = False
    skip_stuck: bool = False

    # GPU opcional. La suite corre en CPU si CuPy/CUDA no está disponible.
    # Valores: auto, on, off. En auto usa GPU solo si detecta CuPy + CUDA.
    gpu: str = "auto"
    gpu_device: int = 0
    gpu_min_elements: int = 250_000

    emg_dirname: str = "EMG"
    dyn_dirname: str = "Dinamometria"
    ct_dirnames: tuple[str, ...] = ("TAC", "Tomografia", "Tomografias", "CT")
    resonance_dirnames: tuple[str, ...] = ("RESONANCIA", "Resonancia", "MRI", "FMRI", "fMRI")
    functional_dirnames: tuple[str, ...] = (
        "ResultadosFuncional", "Resultados funcional", "Resultados Funcional",
        "RESULTADOSFUNCIONAL", "Resultados", "MAPAS", "mapas", "Maps"
    )

    # Derivados externos opcionales. La suite los integra si ya existen; no exige instalar
    # FreeSurfer/CAT12/fMRIPrep para correr el resto.
    derivatives_dirnames: tuple[str, ...] = ("derivatives", "derivados", "DERIVADOS")
    freesurfer_dirnames: tuple[str, ...] = ("freesurfer", "FreeSurfer", "FREESURFER")
    cat12_dirnames: tuple[str, ...] = ("cat12", "CAT12", "spm_cat12", "CAT")
    fmriprep_dirnames: tuple[str, ...] = ("fmriprep", "fMRIprep", "FMRIPREP")

    # Señales
    emg_dt_seconds: float = 0.05
    max_freq_plot_hz: float = 20.0
    signal_entropy_max_scale: int = 12
    signal_cwt_scales: int = 32
    min_samples_signal: int = 20

    # Tomografía
    ct_blind_zone_mm: float = 15.0
    ct_max_muscle_thickness_mm: float = 65.0
    ct_num_rays: int = 60
    ct_hu_muscle_min: float = -50.0
    ct_hu_muscle_max: float = 150.0
    ct_hu_bone_min: float = 300.0
    ct_cwt_min_scale: float = 1.0
    ct_cwt_max_scale: float = 6.0
    ct_cwt_n_scales: int = 12

    # Neuroimagen / mapas
    map_wavelet_scales: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    resonance_wavelet_scales: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    save_histograms: bool = True
    save_mip_previews: bool = True
    nifti_extensions: tuple[str, ...] = (".nii", ".nii.gz")

    # Importante: esta VCE es HEURÍSTICA si no hay DTI/atlas/registro MNI.
    vce_radius_vox: int = 4
    vce_min_z_fraction: float = 0.12
    vce_max_z_fraction: float = 0.92

    # Corteza motora en mapa AD (difusividad axial).
    # Importante: delimitación HEURÍSTICA/exploratoria, no sustituye atlas ni segmentación clínica.
    ad_motor_wavelet_percentile: float = 75.0
    ad_motor_shell_erosion_iters: int = 2
    ad_motor_min_component_voxels: int = 20
    ad_motor_prior_threshold: float = 0.25
    ad_motor_use_atlas_guided: bool = True
    ad_motor_prior_sigma_scale: float = 1.0

    def data_root(self) -> Path:
        return self.data_root_override if self.data_root_override is not None else self.project_root / self.data_dirname

    def results_root(self) -> Path:
        return self.results_root_override if self.results_root_override is not None else self.project_root / self.results_dirname


DEFAULT_PROJECT_ROOT = Path("/home/humath/Escritorio")
