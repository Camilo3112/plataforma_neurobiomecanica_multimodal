"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       MORFOMETRÍA COMPARATIVA EXTERNA                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Archivo: src/external_morphometry.py
Versión: v3.21.21

Descripción
-----------
Aporta funciones auxiliares al pipeline multimodal de análisis
neurobiomecánico.

Fundamento físico-matemático implementado
-----------------------------------------
Compara morfometría del paciente contra referencias externas cuando están
disponibles. El modelo usa normalización por dominio, diferencias respecto a
sano
y proporciones relativas. Las métricas se organizan para no mezclar unidades:
volumen, área, intensidad, porcentaje o densidad permanecen etiquetados.

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

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .paths import resolve_stage_dir, norm_key
from .io_utils import safe_mkdir, save_dataframe, append_log, log_exception, save_json
from .checkpoint import CheckpointManager, load_json_if_exists, save_metrics_json
from .neuroimage import (
    _require_neuro_libs,
    _as_3d,
    save_nifti,
    save_orthogonal_png,
    _voxel_volume_ml_from_affine,
    resize_to_shape,
)

# FreeSurfer Desikan-Killiany aparc+aseg cortical label IDs.
# M1 = precentral. M2/premotora/SMA se aproxima con caudal middle frontal,
# superior frontal y paracentral porque Desikan no trae una etiqueta SMA pura.
FS_LABELS = {
    "m1_izquierda": [1024],
    "m1_derecha": [2024],
    "m2_izquierda": [1003, 1028, 1017],
    "m2_derecha": [2003, 2028, 2017],
}
FS_REGION_LABELS = {
    1024: "precentral_lh",
    2024: "precentral_rh",
    1003: "caudalmiddlefrontal_lh",
    2003: "caudalmiddlefrontal_rh",
    1028: "superiorfrontal_lh",
    2028: "superiorfrontal_rh",
    1017: "paracentral_lh",
    2017: "paracentral_rh",
}


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


def _candidate_roots(cfg: PipelineConfig, subject: str, stage: str) -> list[Path]:
    roots: list[Path] = []
    stage_dir = _subject_stage_dir(cfg, subject, stage)
    if stage_dir:
        roots.append(stage_dir)
        roots.extend([stage_dir / d for d in cfg.derivatives_dirnames])
    roots.extend([cfg.project_root / d for d in cfg.derivatives_dirnames])
    roots.extend([cfg.data_root() / d for d in cfg.derivatives_dirnames])
    roots.append(cfg.results_root() / "derivados_externos")
    # Filtrar existentes, sin duplicar.
    out=[]; seen=set()
    for r in roots:
        try:
            rr=r.resolve()
        except Exception:
            rr=r
        if r.exists() and str(rr) not in seen:
            out.append(r); seen.add(str(rr))
    return out


def _subject_tokens(subject: str, stage: str) -> list[str]:
    s = norm_key(subject)
    compact = re.sub(r"[^a-z0-9]+", "", s)
    stage_norm = norm_key(stage)
    # paciente 1 -> paciente1, paciente_1, sub-paciente1, etc.
    tokens = [s, compact, s.replace(" ", "_"), s.replace(" ", "-"), stage_norm, stage_norm.replace("é", "e")]
    return [t for t in tokens if t]


def _is_probably_subject_stage_path(path: Path, subject: str, stage: str) -> bool:
    n = norm_key(str(path))
    compact = re.sub(r"[^a-z0-9]+", "", n)
    subject_compact = re.sub(r"[^a-z0-9]+", "", norm_key(subject))
    stage_norm = norm_key(stage).replace("é", "e")
    # Si no aparece el sujeto, no descartamos del todo cuando está dentro de la etapa exacta.
    subject_ok = subject_compact in compact or norm_key(subject) in n
    stage_ok = stage_norm in n or (stage_norm == "despues" and "desp" in n) or (stage_norm == "antes" and "antes" in n)
    return subject_ok or stage_ok


def _find_freesurfer_subject_dirs(cfg: PipelineConfig, subject: str, stage: str) -> list[Path]:
    out=[]
    for root in _candidate_roots(cfg, subject, stage):
        # Buscar carpetas que parezcan SUBJECTS_DIR/<subject> con mri/stats.
        for p in root.rglob("*"):
            if not p.is_dir():
                continue
            if (p / "mri").is_dir() and (p / "stats").is_dir():
                has_aparc = any((p / "mri" / name).exists() for name in ["aparc+aseg.mgz", "aparc+aseg.nii", "aparc+aseg.nii.gz"])
                has_stats = (p / "stats" / "lh.aparc.stats").exists() or (p / "stats" / "rh.aparc.stats").exists()
                if (has_aparc or has_stats) and _is_probably_subject_stage_path(p, subject, stage):
                    out.append(p)
        # También raíz directa si ya es un subject dir.
        if (root / "mri").is_dir() and (root / "stats").is_dir():
            out.append(root)
    return sorted(set(out), key=lambda x: len(str(x)))


def _find_first(root: Path, names: list[str]) -> Optional[Path]:
    for name in names:
        p = root / name
        if p.exists():
            return p
    return None


def _parse_aparc_stats(path: Path, hemi: str) -> pd.DataFrame:
    rows=[]
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"):
                continue
            parts=line.split()
            if len(parts) < 5:
                continue
            # StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
            try:
                rows.append({
                    "source": str(path),
                    "tool": "FreeSurfer",
                    "hemi": hemi,
                    "struct_name": parts[0],
                    "num_vert": float(parts[1]),
                    "surface_area_mm2": float(parts[2]),
                    "gray_volume_mm3": float(parts[3]),
                    "thick_avg_mm": float(parts[4]),
                    "thick_std_mm": float(parts[5]) if len(parts) > 5 else math.nan,
                })
            except Exception:
                continue
    return pd.DataFrame(rows)


def _fs_stats_motor_summary(fs_dir: Path) -> pd.DataFrame:
    dfs=[]
    for hemi, fname in [("izquierda", "lh.aparc.stats"), ("derecha", "rh.aparc.stats")]:
        df=_parse_aparc_stats(fs_dir / "stats" / fname, hemi)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    df=pd.concat(dfs, ignore_index=True)
    motor_names={
        "precentral": "m1",
        "caudalmiddlefrontal": "m2",
        "superiorfrontal": "m2",
        "paracentral": "m2",
    }
    rows=[]
    for _, r in df.iterrows():
        region=motor_names.get(str(r["struct_name"]).lower())
        if not region:
            continue
        rows.append({
            "tool": "FreeSurfer",
            "source_type": "stats",
            "region": region,
            "region_detail": r["struct_name"],
            "side": r["hemi"],
            "volume_ml": float(r["gray_volume_mm3"]) / 1000.0,
            "volume_mm3": float(r["gray_volume_mm3"]),
            "surface_area_mm2": r.get("surface_area_mm2", math.nan),
            "thickness_mm": r.get("thick_avg_mm", math.nan),
            "source": r["source"],
            "note": "FreeSurfer aparc.stats; M1=precentral; M2 aproximada=caudalmiddlefrontal/superiorfrontal/paracentral.",
        })
    return pd.DataFrame(rows)


def _process_fs_aparc_masks(cfg: PipelineConfig, fs_dir: Path, out_dir: Path) -> pd.DataFrame:
    _, nib, _ = _require_neuro_libs()
    mri_dir = fs_dir / "mri"
    aparc = _find_first(mri_dir, ["aparc+aseg.mgz", "aparc+aseg.nii.gz", "aparc+aseg.nii"])
    if aparc is None:
        return pd.DataFrame()
    img=nib.load(str(aparc))
    data=_as_3d(img.get_fdata(dtype=np.float32)).astype(np.int32)
    affine=img.affine
    voxel_ml=_voxel_volume_ml_from_affine(affine)
    mask_dir=out_dir / "freesurfer" / "mascaras_motoras"
    safe_mkdir(mask_dir)

    # Máscara general cortical en aparc+aseg.
    cortex=((data >= 1000) & (data < 3000))
    save_nifti(cortex.astype(np.uint8), affine, mask_dir / "cortex_gray_freesurfer_aparc_mask.nii.gz")
    save_orthogonal_png(cortex.astype(np.float32), mask_dir / "cortex_gray_freesurfer_aparc_mask.png", title="FreeSurfer cortex mask")

    labelmap=np.zeros_like(data, dtype=np.uint8)
    rows=[]
    lut_order=[("m1", "primaria", "derecha", FS_LABELS["m1_derecha"], 1),
               ("m1", "primaria", "izquierda", FS_LABELS["m1_izquierda"], 2),
               ("m2", "secundaria", "derecha", FS_LABELS["m2_derecha"], 3),
               ("m2", "secundaria", "izquierda", FS_LABELS["m2_izquierda"], 4)]
    for region, region_name, side, labels, code in lut_order:
        mask=np.isin(data, labels)
        labelmap[mask]=code
        tag=f"freesurfer_corteza_motora_{region_name}_{side}"
        out_mask=mask_dir / f"{tag}.nii.gz"
        save_nifti(mask.astype(np.uint8), affine, out_mask)
        rows.append({
            "tool": "FreeSurfer",
            "source_type": "aparc_aseg_mask",
            "region": region,
            "region_name": region_name,
            "side": side,
            "labels": ",".join(str(x) for x in labels),
            "label_names": ",".join(FS_REGION_LABELS.get(x, str(x)) for x in labels),
            "voxels": int(np.sum(mask)),
            "volume_ml": float(np.sum(mask) * voxel_ml),
            "source": str(aparc),
            "mask_path": str(out_mask),
            "note": "M1=precentral. M2/premotora/SMA aproximada en Desikan por caudalmiddlefrontal+superiorfrontal+paracentral.",
        })
    out_label=mask_dir / "freesurfer_corteza_motora_labelmap.nii.gz"
    save_nifti(labelmap.astype(np.uint8), affine, out_label)
    save_json({"1":"M1 derecha", "2":"M1 izquierda", "3":"M2 derecha", "4":"M2 izquierda"}, mask_dir / "freesurfer_corteza_motora_labelmap_lut.json")
    df=pd.DataFrame(rows)
    if not df.empty:
        save_dataframe(df, mask_dir / "metricas_mascaras_motoras_freesurfer.csv")
    return df


def _process_freesurfer(cfg: PipelineConfig, subject: str, stage: str, out_dir: Path, log_file: Path) -> pd.DataFrame:
    rows=[]
    fs_dirs=_find_freesurfer_subject_dirs(cfg, subject, stage)
    if fs_dirs:
        save_dataframe(pd.DataFrame([{"freesurfer_subject_dir": str(p)} for p in fs_dirs]), out_dir / "freesurfer_dirs_detectados.csv")
    for fs_dir in fs_dirs[:2]:
        try:
            df_masks=_process_fs_aparc_masks(cfg, fs_dir, out_dir)
            df_stats=_fs_stats_motor_summary(fs_dir)
            for df in [df_masks, df_stats]:
                if df is not None and not df.empty:
                    df.insert(0, "stage", stage)
                    df.insert(0, "subject", subject)
                    rows.append(df)
        except Exception as exc:
            log_exception(log_file, f"Integrando FreeSurfer {subject} {stage} {fs_dir}", exc)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _find_cat12_files(cfg: PipelineConfig, subject: str, stage: str) -> list[Path]:
    files=[]
    for root in _candidate_roots(cfg, subject, stage):
        for dname in cfg.cat12_dirnames:
            cat = root / dname
            if cat.exists():
                root2=cat
            else:
                root2=root
            for pat in ["*catROI*.xml", "*catROI*.csv", "*catROIs*.xml", "*cat*.xml", "mwp1*.nii", "mwp1*.nii.gz", "p1*.nii", "p1*.nii.gz"]:
                files.extend([p for p in root2.rglob(pat) if p.is_file() and _is_probably_subject_stage_path(p, subject, stage)])
    return sorted(set(files))


def _try_parse_float(text: str) -> Optional[float]:
    try:
        return float(str(text).strip().replace(",", "."))
    except Exception:
        return None


def _parse_cat12_xml(path: Path) -> pd.DataFrame:
    rows=[]
    try:
        root=ET.parse(path).getroot()
    except Exception:
        return pd.DataFrame()
    # Parser genérico: busca nodos con texto que mencione regiones motoras y captura números cercanos.
    for elem in root.iter():
        txt=" ".join([str(elem.tag), str(elem.attrib), str(elem.text or "")])
        key=norm_key(txt)
        if not any(k in key for k in ["precentral", "supp", "motor", "premotor", "paracentral"]):
            continue
        nums=[]
        for child in list(elem):
            val=_try_parse_float(child.text or "")
            if val is not None:
                nums.append(val)
        for v in nums[:3] or [math.nan]:
            rows.append({
                "tool": "CAT12",
                "source_type": "cat12_xml_roi",
                "region": "m1" if "precentral" in key else ("m2" if any(k in key for k in ["supp", "premotor", "paracentral"]) else "motor"),
                "region_detail": txt[:220],
                "side": "izquierda" if any(k in key for k in ["left", " lh", "izq", "izquierda"]) else ("derecha" if any(k in key for k in ["right", " rh", "der", "derecha"]) else "desconocido"),
                "volume_ml": v / 1000.0 if np.isfinite(v) and v > 100 else v,
                "raw_value": v,
                "source": str(path),
                "note": "Parser genérico CAT12; revisar columnas/unidades del XML original.",
            })
    return pd.DataFrame(rows)


def _process_cat12(cfg: PipelineConfig, subject: str, stage: str, out_dir: Path, log_file: Path) -> pd.DataFrame:
    _, nib, _ = _require_neuro_libs()
    files=_find_cat12_files(cfg, subject, stage)
    if files:
        save_dataframe(pd.DataFrame([{"cat12_file": str(p), "name": p.name} for p in files]), out_dir / "cat12_archivos_detectados.csv")
    rows=[]
    gm_dir=out_dir / "cat12"
    for p in files:
        try:
            name=p.name.lower()
            if name.endswith(".xml"):
                df=_parse_cat12_xml(p)
                if not df.empty:
                    df.insert(0,"stage",stage); df.insert(0,"subject",subject); rows.append(df)
            elif name.endswith(".nii") or name.endswith(".nii.gz"):
                img=nib.load(str(p)); data=_as_3d(img.get_fdata(dtype=np.float32)); affine=img.affine
                base=re.sub(r"[^A-Za-z0-9_]+","_",p.name.replace(".nii.gz","").replace(".nii",""))
                out_nii=gm_dir / f"{base}_cat12_gray_matter_prob.nii.gz"
                save_nifti(data.astype(np.float32), affine, out_nii)
                # máscara GM sensible, útil para QC pero no específica de M1/M2.
                mask=data > np.nanpercentile(data[data>0], 70) if np.any(data>0) else data>0
                out_mask=gm_dir / f"{base}_cat12_gray_matter_mask_p70.nii.gz"
                save_nifti(mask.astype(np.uint8), affine, out_mask)
                rows.append(pd.DataFrame([{
                    "subject":subject,"stage":stage,"tool":"CAT12","source_type":"gray_matter_map",
                    "region":"gray_matter_global","side":"desconocido","voxels":int(np.sum(mask)),
                    "volume_ml":float(np.sum(mask)*_voxel_volume_ml_from_affine(affine)),"source":str(p),
                    "mask_path":str(out_mask),"note":"Mapa p1/mwp1 CAT12; no específico de M1/M2 sin atlas ROI adicional."
                }]))
        except Exception as exc:
            log_exception(log_file, f"Integrando CAT12 {subject} {stage} {p}", exc)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _find_fmriprep_files(cfg: PipelineConfig, subject: str, stage: str) -> list[Path]:
    files=[]
    for root in _candidate_roots(cfg, subject, stage):
        for dname in cfg.fmriprep_dirnames:
            fp = root / dname
            root2 = fp if fp.exists() else root
            for pat in ["*desc-confounds_timeseries.tsv", "*confounds*.tsv", "*desc-preproc_bold.nii.gz", "*boldref.nii.gz", "*desc-brain_mask.nii.gz"]:
                files.extend([p for p in root2.rglob(pat) if p.is_file() and _is_probably_subject_stage_path(p, subject, stage)])
    return sorted(set(files))


def _process_fmriprep(cfg: PipelineConfig, subject: str, stage: str, out_dir: Path, log_file: Path) -> pd.DataFrame:
    files=_find_fmriprep_files(cfg, subject, stage)
    if files:
        save_dataframe(pd.DataFrame([{"fmriprep_file": str(p), "name": p.name} for p in files]), out_dir / "fmriprep_archivos_detectados.csv")
    rows=[]
    for p in files:
        try:
            if p.suffix.lower() == ".tsv":
                df=pd.read_csv(p, sep="\t")
                row={"subject":subject,"stage":stage,"tool":"fMRIPrep","source_type":"confounds_tsv","source":str(p),"n_timepoints":int(len(df))}
                for col in df.columns:
                    key=norm_key(col)
                    if key in {"framewise_displacement", "std_dvars", "dvars", "global_signal", "csf", "white_matter"} or "trans_" in key or "rot_" in key:
                        vals=pd.to_numeric(df[col], errors="coerce")
                        row[f"{col}_mean"] = float(vals.mean(skipna=True))
                        row[f"{col}_p95"] = float(vals.quantile(0.95))
                        row[f"{col}_max"] = float(vals.max(skipna=True))
                rows.append(row)
            else:
                rows.append({"subject":subject,"stage":stage,"tool":"fMRIPrep","source_type":"nifti_or_report","source":str(p),"note":"Archivo fMRIPrep detectado para posible coregistro/QC."})
        except Exception as exc:
            log_exception(log_file, f"Integrando fMRIPrep {subject} {stage} {p}", exc)
    return pd.DataFrame(rows)




def _stage_label(stage: str) -> str:
    return norm_key(stage).replace(" ", "_").replace("é", "e")


def _subject_label(subject: str) -> str:
    return norm_key(subject).replace(" ", "_").replace("-", "_")


def prepare_derivatives_workspace(cfg: PipelineConfig) -> Path:
    """Crea una estructura de trabajo para derivados externos.

    FreeSurfer, CAT12 y fMRIPrep NO se calculan dentro de esta suite. Esta función
    solo crea carpetas limpias para que el usuario pegue ahí las salidas cuando las tenga.
    Si no hay derivados, el pipeline continúa con el método heurístico interno.
    """
    base = cfg.results_root() / "derivados_externos"
    safe_mkdir(base)
    subjects = list(cfg.patients) + [cfg.control_name]
    expected_rows = []
    for subject in subjects:
        stages = list(cfg.stages)
        if subject == cfg.control_name:
            stages = ["Antes"]
        for stage in stages:
            folder = f"{_subject_label(subject)}_{_stage_label(stage)}"
            fs_dir = base / "freesurfer" / folder
            cat_dir = base / "cat12" / folder
            fp_dir = base / "fmriprep" / folder
            for d in [fs_dir / "mri", fs_dir / "stats", cat_dir, fp_dir / "func"]:
                safe_mkdir(d)
            expected_rows.extend([
                {"subject": subject, "stage": stage, "tool": "FreeSurfer", "expected_path": str(fs_dir / "mri" / "aparc+aseg.mgz"), "required": "sí para máscaras anatómicas M1/M2"},
                {"subject": subject, "stage": stage, "tool": "FreeSurfer", "expected_path": str(fs_dir / "stats" / "lh.aparc.stats"), "required": "sí para volumen/grosor izquierdo"},
                {"subject": subject, "stage": stage, "tool": "FreeSurfer", "expected_path": str(fs_dir / "stats" / "rh.aparc.stats"), "required": "sí para volumen/grosor derecho"},
                {"subject": subject, "stage": stage, "tool": "CAT12", "expected_path": str(cat_dir / "catROI_*.xml"), "required": "opcional"},
                {"subject": subject, "stage": stage, "tool": "CAT12", "expected_path": str(cat_dir / "mwp1*.nii"), "required": "opcional"},
                {"subject": subject, "stage": stage, "tool": "fMRIPrep", "expected_path": str(fp_dir / "func" / "*desc-confounds_timeseries.tsv"), "required": "opcional QC fMRI"},
            ])
    df = pd.DataFrame(expected_rows)
    save_dataframe(df, base / "plantilla_rutas_derivados_esperadas.csv")
    (base / "LEEME_DERIVADOS_EXTERNOS.txt").write_text(
        "Estas carpetas son una plantilla creada automáticamente.\n\n"
        "IMPORTANTE: FreeSurfer, CAT12 y fMRIPrep no vienen calculados en tus datos actuales.\n"
        "Debes pegar aquí sus salidas si las generas por fuera. Si no las tienes, la suite sigue funcionando con el método heurístico interno.\n\n"
        "FreeSurfer mínimo por sujeto/etapa:\n"
        "  freesurfer/<sujeto_etapa>/mri/aparc+aseg.mgz\n"
        "  freesurfer/<sujeto_etapa>/stats/lh.aparc.stats\n"
        "  freesurfer/<sujeto_etapa>/stats/rh.aparc.stats\n\n"
        "CAT12 opcional:\n"
        "  cat12/<sujeto_etapa>/catROI*.xml\n"
        "  cat12/<sujeto_etapa>/mwp1*.nii\n\n"
        "fMRIPrep opcional:\n"
        "  fmriprep/<sujeto_etapa>/func/*desc-confounds_timeseries.tsv\n",
        encoding="utf-8",
    )
    return base


def _write_derivatives_instructions(cfg: PipelineConfig) -> Path:
    out=cfg.results_root() / "_instrucciones_derivados_externos.md"
    safe_mkdir(out.parent)
    out.write_text(
        "# Cómo anexar FreeSurfer/CAT12/fMRIPrep a la suite\n\n"
        "La suite v3.9 integra estos derivados si ya están calculados. No los calcula internamente en Windows porque FreeSurfer/fMRIPrep suelen correr mejor en WSL2/Docker/Linux y CAT12 en MATLAB/SPM.\n\n"
        "## Estructura sugerida\n\n"
        "```text\n"
        "D:/EAFIT/01-2026/proyecto/derivatives/\n"
        "  freesurfer/\n"
        "    paciente_1_Antes/\n"
        "      mri/aparc+aseg.mgz\n"
        "      stats/lh.aparc.stats\n"
        "      stats/rh.aparc.stats\n"
        "    paciente_1_Despues/ ...\n"
        "    paciente_2_Antes/ ...\n"
        "    sano_Antes/ ...\n"
        "  cat12/\n"
        "    paciente_1_Antes/catROI_*.xml\n"
        "    paciente_1_Antes/mwp1*.nii\n"
        "  fmriprep/\n"
        "    sub-*/ses-*/func/*desc-confounds_timeseries.tsv\n"
        "```\n\n"
        "## Qué usa la suite\n"
        "- FreeSurfer: `aparc+aseg.mgz`, `lh.aparc.stats`, `rh.aparc.stats`.\n"
        "- CAT12: `catROI*.xml`, `p1*.nii`, `mwp1*.nii`.\n"
        "- fMRIPrep: confounds TSV para FD/DVARS/movimiento y NIfTI preprocesados como referencia.\n\n"
        "## Comando de integración\n"
        "```bat\npython main.py --project-root \"D:\\EAFIT\\01-2026\\proyecto\" --only derivados estructura_funcion correlaciones comparacion --gpu auto --force\n```\n",
        encoding="utf-8",
    )
    return out


def run_external_morphometry_integration(cfg: PipelineConfig) -> pd.DataFrame:
    """Integra derivados externos de FreeSurfer, CAT12 y fMRIPrep si existen.

    Si las carpetas no existen, crea una plantilla en resultados/derivados_externos
    y continúa sin bloquear el resto del pipeline.
    """
    workspace = prepare_derivatives_workspace(cfg)
    all_rows=[]
    ckpt=CheckpointManager(cfg)
    subjects=list(cfg.patients)+[cfg.control_name]
    for subject in subjects:
        stages=cfg.stages
        if subject == cfg.control_name:
            stages=tuple(["Antes"] + [s for s in cfg.stages if s != "Antes" and _subject_stage_dir(cfg, subject, s) is not None])
        for stage in stages:
            print(f"\n[DERIVADOS EXTERNOS] {subject} · {stage}")
            out_dir=cfg.results_root()/subject/stage/"morfometria"
            log_file=cfg.results_root()/subject/stage/"reportes"/"derivados_externos_log.txt"
            safe_mkdir(out_dir)
            task_id=f"derivados_externos/{subject}/{stage}"
            inputs=[]
            for root in _candidate_roots(cfg, subject, stage):
                inputs.append(root)

            def _work(subject=subject, stage=stage, out_dir=out_dir, log_file=log_file):
                dfs=[]
                for fn in [_process_freesurfer, _process_cat12, _process_fmriprep]:
                    try:
                        df=fn(cfg, subject, stage, out_dir, log_file)
                        if df is not None and not df.empty:
                            dfs.append(df)
                    except Exception as exc:
                        log_exception(log_file, f"Integración derivados {fn.__name__} {subject} {stage}", exc)
                df=pd.concat(dfs, ignore_index=True, sort=False) if dfs else pd.DataFrame()
                if not df.empty:
                    save_dataframe(df, out_dir / "resumen_morfometria_externa.csv")
                    try:
                        save_dataframe(df, out_dir / "resumen_morfometria_externa.xlsx")
                    except Exception:
                        pass
                return {"rows": int(len(df)), "out_csv": str(out_dir / "resumen_morfometria_externa.csv") if not df.empty else ""}

            try:
                result,status=ckpt.run(
                    task_id=task_id,
                    inputs=inputs,
                    outputs=[out_dir/"resumen_morfometria_externa.csv"],
                    params={"v":"3_9_external_morphometry"},
                    fn=_work,
                )
                csv=out_dir/"resumen_morfometria_externa.csv"
                if csv.exists():
                    try:
                        df=pd.read_csv(csv)
                        if not df.empty:
                            df["checkpoint_status"]=status
                            all_rows.append(df)
                    except Exception:
                        pass
            except Exception as exc:
                log_exception(log_file, f"Derivados externos global {subject} {stage}", exc)
    _write_derivatives_instructions(cfg)
    out=pd.concat(all_rows, ignore_index=True, sort=False) if all_rows else pd.DataFrame()
    if not out.empty:
        save_dataframe(out, cfg.results_root()/"resumen_global_morfometria_externa.csv")
        try:
            save_dataframe(out, cfg.results_root()/"resumen_global_morfometria_externa.xlsx")
        except Exception:
            pass
    else:
        status = pd.DataFrame([{
            "estado": "sin_derivados_externos_detectados",
            "detalle": "No se encontraron salidas reales de FreeSurfer/CAT12/fMRIPrep. Se creó plantilla y la suite puede seguir con el método heurístico interno.",
            "plantilla": str(cfg.results_root() / "derivados_externos"),
        }])
        save_dataframe(status, cfg.results_root()/"resumen_global_morfometria_externa.csv")
    return out
