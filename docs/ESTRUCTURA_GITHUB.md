# Estructura recomendada para GitHub

```text
suite-integrada-vce/
├── main.py                          # Entrada única del pipeline
├── requirements.txt                  # Dependencias Python
├── pyproject.toml                    # Metadatos básicos del proyecto
├── .gitignore                        # Evita subir datos pesados y resultados
├── config/
│   └── vce_config.example.json       # Plantilla de configuración local
├── docs/
│   ├── USO_MAIN.md
│   ├── MODELOS_FISICOS_MATEMATICOS.md
│   ├── ESTRUCTURA_GITHUB.md
│   └── VISUALIZACION_SLICER_ITKSNAP.md
├── src/                              # Código propio organizado por módulos
│   ├── config.py
│   ├── paths.py
│   ├── brainstem_freesurfer.py
│   ├── cst_tronco_runner.py
│   ├── tomography.py
│   ├── neuroimage.py
│   └── ...
├── tools/                            # Herramientas auxiliares ejecutables
├── modulos/                          # Módulos específicos integrados
└── notebooks/                        # Exploración/estadística si aplica
```

## No subir al repositorio

- `datos/`
- `resultados/`
- `.venv/`
- archivos DICOM/NIfTI/TRK reales de pacientes
- licencia FreeSurfer (`license.txt`)
- archivos temporales de FreeSurfer (`freesurfer_subjects/`, `freesurfer_stage/`)

## Comandos Git básicos

```bash
git init
git add main.py src tools modulos docs config requirements.txt pyproject.toml .gitignore README.md
git commit -m "Organiza suite VCE con main Python y documentación técnica"
git branch -M main
git remote add origin https://github.com/USUARIO/NOMBRE_REPO.git
git push -u origin main
```
