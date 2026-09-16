"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  DERIVADAS Y CAMBIOS LONGITUDINALES REALES                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/real_derivatives.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.

Fundamento físico-matemático implementado
-----------------------------------------
Calcula derivadas y cambios reales entre mediciones del paciente. El operador
principal es la diferencia discreta Δx entre etapas y dominios, con
normalización
por magnitud basal cuando es interpretable. En mapas espaciales, las
diferencias
se calculan voxel a voxel manteniendo misma grilla anatómica.

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

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .paths import resolve_stage_dir, norm_key
from .io_utils import safe_mkdir, save_dataframe
from .neuroimage import (
    collect_dicom_series,
    dicom_series_to_volume,
    classify_series,
    sanitize_name,
    save_nifti,
)


def _subject_stage_dir(cfg: PipelineConfig, subject: str, stage: str) -> Optional[Path]:
    subject_dir = cfg.data_root() / subject
    if subject == cfg.control_name:
        exact = resolve_stage_dir(subject_dir, stage, fallback_to_subject=False)
        if exact is not None:
            return exact
        before = resolve_stage_dir(subject_dir, "Antes", fallback_to_subject=False)
        if before is not None:
            return before
        return subject_dir if subject_dir.exists() else None
    return resolve_stage_dir(subject_dir, stage, fallback_to_subject=False)


def _first_existing_resonance_dir(stage_dir: Optional[Path], cfg: PipelineConfig) -> Optional[Path]:
    if stage_dir is None or not stage_dir.exists():
        return None
    for name in cfg.resonance_dirnames:
        direct = stage_dir / name
        if direct.exists():
            return direct
        wanted = norm_key(name)
        for child in stage_dir.iterdir():
            if child.is_dir() and norm_key(child.name) == wanted:
                return child
    # respaldo: cualquier carpeta que parezca resonancia
    for child in stage_dir.rglob("*"):
        if child.is_dir() and norm_key(child.name) in {"resonancia", "resonancias", "mri", "fmri"}:
            return child
    return None


def bids_subject_label(subject: str) -> str:
    n = norm_key(subject)
    n = n.replace("paciente ", "paciente")
    n = re.sub(r"[^a-z0-9]+", "", n)
    return n or "sujeto"


def bids_session_label(stage: str) -> str:
    n = norm_key(stage).replace("después", "despues").replace("despues", "despues")
    n = re.sub(r"[^a-z0-9]+", "", n)
    return n or "sesion"


def fs_subject_label(subject: str, stage: str) -> str:
    return f"{bids_subject_label(subject)}_{bids_session_label(stage)}"


def _dicom_file_path(ds) -> str:
    return str(getattr(ds, "filename", ""))


def _series_desc(ds) -> str:
    return str(getattr(ds, "SeriesDescription", ""))


def _series_protocol(ds) -> str:
    return str(getattr(ds, "ProtocolName", ""))


def _series_score_for_t1(ds_list: list) -> tuple[int, int, int]:
    if not ds_list:
        return (0, 0, 0)
    ds = ds_list[0]
    cls = classify_series(_series_desc(ds), _series_protocol(ds))
    desc = norm_key(_series_desc(ds) + " " + _series_protocol(ds))
    is_t1 = int(cls.get("series_role") == "anatomica" and cls.get("series_subtype") == "t1")
    is_vol = int("spc" in desc or "mpr" in desc or "vol" in desc)
    return (is_t1, is_vol, len(ds_list))


def _series_score_for_bold(ds_list: list) -> tuple[int, int, int]:
    if not ds_list:
        return (0, 0, 0)
    ds = ds_list[0]
    cls = classify_series(_series_desc(ds), _series_protocol(ds))
    txt = norm_key(_series_desc(ds) + " " + _series_protocol(ds))
    role_ok = int(cls.get("series_role") == "fmri_motor")
    raw_ok = int(cls.get("series_subtype") in {"bold_raw", "moco"} or "ep2d" in txt)
    return (role_ok, raw_ok, len(ds_list))


def build_real_derivatives_workspace(cfg: PipelineConfig, convert_t1: bool = True) -> Path:
    """Prepara BIDS mínimo y scripts para generar derivados reales sin Docker y sin MATLAB.

    Esta función:
    - localiza la mejor serie T1 por sujeto/etapa;
    - exporta un T1w NIfTI mínimo para FreeSurfer nativo;
    - crea scripts Linux para recon-all nativo;
    - crea script opcional para fMRIPrep bare-metal si existe.
    """
    base = cfg.results_root() / "derivados_externos"
    bids = base / "bids"
    fs_out = base / "freesurfer"
    fmriprep_out = base / "fmriprep"
    logs = base / "_logs"
    for d in [bids, fs_out, fmriprep_out, logs]:
        safe_mkdir(d)

    (bids / "dataset_description.json").write_text(json.dumps({
        "Name": "EAFIT VCE proyecto - BIDS minimo generado por suite_integrada_vce",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "GeneratedBy": [{"Name": "suite_integrada_vce", "Version": "v3.16-linux-consolidado"}],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    dcm2niix_rows = []
    subjects = list(cfg.patients) + [cfg.control_name]
    for subject in subjects:
        stages = list(cfg.stages)
        if subject == cfg.control_name:
            # Sano completo: si existen Antes y Despues, ambos se exportan a BIDS.
            stages = [st for st in cfg.stages if resolve_stage_dir(cfg.data_root() / cfg.control_name, st, fallback_to_subject=False) is not None]
            if not stages and (cfg.data_root() / cfg.control_name).exists():
                stages = ["Antes"]
        for stage in stages:
            stage_dir = _subject_stage_dir(cfg, subject, stage)
            res_dir = _first_existing_resonance_dir(stage_dir, cfg)
            row_base = {"subject": subject, "stage": stage, "stage_dir": str(stage_dir) if stage_dir else "", "resonance_dir": str(res_dir) if res_dir else ""}
            if not res_dir:
                rows.append({**row_base, "status": "sin_carpeta_resonancia"})
                continue
            try:
                series = collect_dicom_series(res_dir)
            except Exception as exc:
                rows.append({**row_base, "status": "error_leyendo_dicom", "error": str(exc)})
                continue
            if not series:
                rows.append({**row_base, "status": "sin_series_dicom_pixeldata"})
                continue

            manifest = []
            for uid, ds_list in series.items():
                if not ds_list:
                    continue
                ds = ds_list[0]
                cls = classify_series(_series_desc(ds), _series_protocol(ds))
                files = [Path(_dicom_file_path(d)) for d in ds_list if _dicom_file_path(d)]
                series_folder = str(files[0].parent) if files else ""
                manifest.append({
                    **row_base,
                    "series_uid": uid,
                    "descripcion": _series_desc(ds),
                    "protocolo": _series_protocol(ds),
                    "series_role": cls.get("series_role", ""),
                    "series_subtype": cls.get("series_subtype", ""),
                    "series_side": cls.get("series_side", ""),
                    "n_dicoms": len(ds_list),
                    "series_folder": series_folder,
                })
            if manifest:
                save_dataframe(pd.DataFrame(manifest), logs / f"manifest_series_{fs_subject_label(subject, stage)}.csv")

            t1_candidates = sorted(series.items(), key=lambda kv: _series_score_for_t1(kv[1]), reverse=True)
            best_uid, best_ds = t1_candidates[0]
            best_score = _series_score_for_t1(best_ds)
            if best_score[0] == 0:
                rows.append({**row_base, "status": "sin_t1_claro", "nota": "No encontré serie clasificada como T1; revisar manifest_series."})
            else:
                sub = bids_subject_label(subject)
                ses = bids_session_label(stage)
                anat_dir = bids / f"sub-{sub}" / f"ses-{ses}" / "anat"
                safe_mkdir(anat_dir)
                t1_out = anat_dir / f"sub-{sub}_ses-{ses}_T1w.nii.gz"
                try:
                    if convert_t1 and (cfg.force or not t1_out.exists()):
                        vol, affine, meta = dicom_series_to_volume(best_ds)
                        save_nifti(vol, affine, t1_out)
                        (anat_dir / f"sub-{sub}_ses-{ses}_T1w.json").write_text(json.dumps({
                            "Modality": "MR",
                            "SeriesDescription": meta.get("series_description", ""),
                            "ProtocolName": meta.get("protocol", ""),
                            "MagneticFieldStrength": meta.get("magnetic_field", ""),
                            "RepetitionTime": meta.get("tr", ""),
                            "EchoTime": meta.get("te", ""),
                            "SourceSeriesInstanceUID": best_uid,
                            "GeneratedBy": "suite_integrada_vce conversion DICOM->NIfTI; revisar orientación antes de uso clínico",
                        }, indent=2, ensure_ascii=False), encoding="utf-8")
                    rows.append({**row_base, "status": "t1_exportado", "bids_t1w": str(t1_out), "series_uid": best_uid, "score": str(best_score)})
                except Exception as exc:
                    rows.append({**row_base, "status": "error_exportando_t1", "series_uid": best_uid, "error": str(exc)})

            # Registrar series fMRI motor candidatas para convertir con dcm2niix si se quiere fMRIPrep completo.
            bold_candidates = sorted(series.items(), key=lambda kv: _series_score_for_bold(kv[1]), reverse=True)
            for uid, ds_list in bold_candidates[:4]:
                score = _series_score_for_bold(ds_list)
                if score[0] == 0:
                    continue
                ds = ds_list[0]
                cls = classify_series(_series_desc(ds), _series_protocol(ds))
                files = [Path(_dicom_file_path(d)) for d in ds_list if _dicom_file_path(d)]
                if not files:
                    continue
                sub = bids_subject_label(subject)
                ses = bids_session_label(stage)
                func_dir = bids / f"sub-{sub}" / f"ses-{ses}" / "func"
                safe_mkdir(func_dir)
                side = cls.get("series_side", "desconocido")
                task = "motorder" if side == "derecha" else "motorizq" if side == "izquierda" else "motor"
                dcm2niix_rows.append({
                    "subject": subject,
                    "stage": stage,
                    "series_uid": uid,
                    "descripcion": _series_desc(ds),
                    "protocolo": _series_protocol(ds),
                    "side": side,
                    "dicom_folder": str(files[0].parent),
                    "suggested_output_folder": str(func_dir),
                    "suggested_bids_filename": f"sub-{sub}_ses-{ses}_task-{task}_bold",
                    "nota": "Convertir con dcm2niix para fMRIPrep completo; la conversión de fMRI Siemens debe hacerse con dcm2niix, no con el conversor simple de la suite.",
                })

    df = pd.DataFrame(rows)
    save_dataframe(df, base / "t1_bids_exportados.csv")
    if dcm2niix_rows:
        save_dataframe(pd.DataFrame(dcm2niix_rows), base / "series_fmri_para_dcm2niix.csv")
    _write_derivative_scripts(cfg, base, df, pd.DataFrame(dcm2niix_rows))
    return base


def _win(path: Path) -> str:
    return str(path).replace("/", "\\")


def _sh(path: Path) -> str:
    return str(path)


def _write_derivative_scripts(cfg: PipelineConfig, base: Path, t1_df: pd.DataFrame, bold_df: pd.DataFrame) -> None:
    """Crea scripts para Linux sin Docker y sin MATLAB.

    Esta versión prepara:
    - FreeSurfer nativo con recon-all;
    - dcm2niix opcional para BOLD;
    - fMRIPrep bare-metal opcional si el comando existe;
    - integración posterior a la suite.
    """
    bids = base / "bids"
    fs_out = base / "freesurfer"
    fmriprep_out = base / "fmriprep"
    license_path = cfg.project_root / "license.txt"
    suite_dir = Path(__file__).resolve().parents[1]

    labels = sorted(set(bids_subject_label(str(r.get("subject", ""))) for _, r in t1_df.iterrows() if r.get("status") == "t1_exportado")) if not t1_df.empty else []
    label_args = " ".join(labels)

    # Linux nativo: FreeSurfer.
    fs_sh = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "echo '============================================================'",
        "echo 'FreeSurfer nativo - recon-all sin Docker y sin MATLAB'",
        "echo '============================================================'",
        f"PROJECT_ROOT='{_sh(cfg.project_root)}'",
        f"BIDS='{_sh(bids)}'",
        f"SUBJECTS_DIR='{_sh(fs_out)}'",
        f"FS_LICENSE='{_sh(license_path)}'",
        "if [ ! -f \"$FS_LICENSE\" ]; then",
        "  echo \"Falta license.txt de FreeSurfer en: $FS_LICENSE\"",
        "  echo \"Revisa GUIA_LICENSE_FREESURFER.md\"",
        "  exit 1",
        "fi",
        "export FS_LICENSE",
        "mkdir -p \"$SUBJECTS_DIR\"",
        "",
        "# Buscar FreeSurfer nativo si recon-all todavía no está en PATH.",
        "if ! command -v recon-all >/dev/null 2>&1; then",
        "  FS_SETUP=\"\"",
        "  for CAND in \\",
        "    /usr/local/freesurfer/8.0.0/SetUpFreeSurfer.sh \\",
        "    /usr/local/freesurfer/7.4.1/SetUpFreeSurfer.sh \\",
        "    /usr/local/freesurfer/7.3.2/SetUpFreeSurfer.sh \\",
        "    /usr/local/freesurfer/SetUpFreeSurfer.sh \\",
        "    $HOME/freesurfer/SetUpFreeSurfer.sh; do",
        "    if [ -f \"$CAND\" ]; then FS_SETUP=\"$CAND\"; break; fi",
        "  done",
        "  if [ -n \"$FS_SETUP\" ]; then",
        "    echo \"Activando FreeSurfer: $FS_SETUP\"",
        "    # shellcheck disable=SC1090",
        "    source \"$FS_SETUP\"",
        "  fi",
        "fi",
        "command -v recon-all >/dev/null 2>&1 || { echo 'No encontré recon-all. Instala FreeSurfer nativo.'; exit 1; }",
        "export SUBJECTS_DIR",
    ]
    for _, r in t1_df.iterrows() if not t1_df.empty else []:
        if r.get("status") != "t1_exportado":
            continue
        subject = str(r.get("subject", "")); stage = str(r.get("stage", ""))
        sub = bids_subject_label(subject); ses = bids_session_label(stage); fsid = fs_subject_label(subject, stage)
        t1_path = bids / f"sub-{sub}" / f"ses-{ses}" / "anat" / f"sub-{sub}_ses-{ses}_T1w.nii.gz"
        fs_sh += [
            "",
            "echo '------------------------------------------------------------'",
            f"echo 'Procesando FreeSurfer nativo: {fsid}'",
            f"echo 'T1: {_sh(t1_path)}'",
            "echo '------------------------------------------------------------'",
            f"if [ -f \"$SUBJECTS_DIR/{fsid}/scripts/recon-all.done\" ]; then",
            f"  echo 'Ya existe recon-all.done para {fsid}; saltando.'",
            "else",
            f"  recon-all -i '{_sh(t1_path)}' -s '{fsid}' -sd \"$SUBJECTS_DIR\" -all",
            "fi",
        ]
    fs_sh += ["echo 'FreeSurfer nativo terminado.'"]
    (base / "run_02_freesurfer_recon_all_nativo.sh").write_text("\n".join(fs_sh) + "\n", encoding="utf-8")

    # dcm2niix opcional para BOLD.
    dcm_sh = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "command -v dcm2niix >/dev/null 2>&1 || { echo 'No encuentro dcm2niix. Instálalo con sudo apt install dcm2niix'; exit 1; }",
    ]
    if not bold_df.empty:
        for _, r in bold_df.iterrows():
            out_b = str(r.get("suggested_output_folder", "")); fn = str(r.get("suggested_bids_filename", "")); folder = str(r.get("dicom_folder", ""))
            dcm_sh += [
                f"mkdir -p '{out_b}'",
                f"dcm2niix -z y -b y -ba n -f '{fn}' -o '{out_b}' '{folder}'",
            ]
    else:
        dcm_sh.append("echo 'No se detectaron series fMRI motor candidatas en el manifest.'")
    (base / "run_01_convertir_fmri_dcm2niix_opcional.sh").write_text("\n".join(dcm_sh) + "\n", encoding="utf-8")

    # fMRIPrep bare-metal opcional.
    fmr_sh = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"BIDS='{_sh(bids)}'",
        f"OUT='{_sh(fmriprep_out)}'",
        f"FS_LICENSE='{_sh(license_path)}'",
        "if [ ! -f \"$FS_LICENSE\" ]; then echo \"Falta license.txt: $FS_LICENSE\"; exit 1; fi",
        "command -v fmriprep >/dev/null 2>&1 || { echo 'No encontré fmriprep bare-metal. Puedes omitirlo y usar ResultadosFuncional.'; exit 0; }",
        "mkdir -p \"$OUT\"",
        f"fmriprep \"$BIDS\" \"$OUT\" participant --anat-only --fs-license-file \"$FS_LICENSE\" --participant-label {label_args}",
    ]
    (base / "run_03_fmriprep_anat_only_baremetal_opcional.sh").write_text("\n".join(fmr_sh) + "\n", encoding="utf-8")

    # Integración.
    integration_sh = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd '{_sh(suite_dir)}'",
        "source .venv/bin/activate 2>/dev/null || true",
        f"python3 main.py --project-root '{_sh(cfg.project_root)}' --data-root '{_sh(cfg.data_root())}' --only derivados estructura_funcion correlaciones comparacion --gpu auto --force",
    ]
    (base / "run_06_integrar_derivados_en_suite.sh").write_text("\n".join(integration_sh) + "\n", encoding="utf-8")

    for sh in base.glob('run_*.sh'):
        try:
            sh.chmod(0o755)
        except Exception:
            pass

    readme = f"""
# Derivados reales sin Docker y sin MATLAB

Esta carpeta fue generada por `suite_integrada_vce v3.15-linux-sin-matlab`.

## Qué se usa

- FreeSurfer nativo para morfometría real.
- dcm2niix opcional para BOLD.
- fMRIPrep bare-metal opcional si está instalado.
- No usa Docker.
- No usa MATLAB.
- No usa CAT12.

## Orden recomendado

1. Revisar T1 exportados:
   - `{bids}`
   - archivo resumen: `t1_bids_exportados.csv`

2. Correr FreeSurfer real nativo:
   - `run_02_freesurfer_recon_all_nativo.sh`
   - requiere FreeSurfer instalado y `license.txt` en:
     `{license_path}`

3. Opcional para fMRI BIDS:
   - `run_01_convertir_fmri_dcm2niix_opcional.sh`

4. Opcional si instalaste fMRIPrep bare-metal:
   - `run_03_fmriprep_anat_only_baremetal_opcional.sh`

5. Integrar todo a la suite:
   - `run_06_integrar_derivados_en_suite.sh`

## Nota

La morfometría fuerte sale de FreeSurfer nativo. Si fMRIPrep no está disponible, la suite sigue usando tus mapas `ResultadosFuncional` y los cruza con FreeSurfer/T1/AD.
"""
    (base / "README_DERIVADOS_REALES.md").write_text(readme, encoding="utf-8")

def run_real_derivatives_preparation(cfg: PipelineConfig) -> Path:
    return build_real_derivatives_workspace(cfg, convert_t1=True)
