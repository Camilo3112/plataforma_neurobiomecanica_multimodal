# Suite VCE v3.21.6 Linux — QC dos fémures + CST/tronco integrado

Esta versión lleva a Linux la corrección de tomografía y la integración del módulo CST/tronco.

## Cambios principales

1. **Tomografía con control de dos fémures**: antes de calcular áreas/volúmenes, el sistema escanea el stack DICOM y verifica que el corte elegido tenga evidencia de dos fémures. Si el corte medio cae en un solo fémur, no duplica resultados; busca el corte bilateral más cercano o, si es necesario, rescata un corte bilateral en todo el volumen.

2. **Segmentación completa del miembro**: se evita cortar un muslo por la mitad usando una línea media global. El procesamiento prioriza el contorno anatómico alrededor del fémur y deja advertencias cuando el campo de visión esté truncado.

3. **CST/tronco en Linux nativo**: el runner puede ejecutar FreeSurfer directamente en Linux para segmentar mesencéfalo, puente, bulbo y pedúnculo cerebeloso superior. Luego valida las máscaras en espacio rT1 y ejecuta la visualización CST ventral/medial si existe `wholebrain_T1.trk`.

## Rutas esperadas

- Proyecto: `/home/humath/Escritorio`
- Datos: `/home/humath/Escritorio/datos`
- Resultados: `/home/humath/Escritorio/resultados`
- Licencia FreeSurfer: `/home/humath/Escritorio/license.txt`

## Instalación rápida

```bash
cd /home/humath/Escritorio/suite_integrada_vce_v3_21_6_linux_qc_rescate_cst_integrado
chmod +x *.sh
find modulos/modulo_CST_tronco_integracion_completa -name "*.sh" -exec chmod +x {} \;
./setup_linux_v3_21_6.sh
```

## Reprocesar TAC pacientes 3 y 6

```bash
./run_reprocesar_tomografia_qc_dos_femures_pacientes_3_6_linux.sh
./run_consolidar_pacientes_3_6_linux.sh
```

## Ejecutar CST/tronco

Primero instala/carga FreeSurfer y guarda la licencia:

```bash
cp /ruta/a/license.txt /home/humath/Escritorio/license.txt
./instalar_freesurfer_nativo_linux.sh
source /usr/local/freesurfer/SetUpFreeSurfer.sh
export FS_LICENSE=/home/humath/Escritorio/license.txt
```

Luego ejecuta:

```bash
./run_cst_tronco_paciente6_antes_linux.sh
```

El CST solo corre si ya existe:

```text
/home/humath/Escritorio/resultados/paciente 6/Antes/tractografia_propia/wholebrain_T1.trk
```

Si no existe, el módulo quedará omitido con estado `omitido_sin_wholebrain_T1_trk`.

## Archivos QC importantes

En cada salida de tomografía avanzada revisa:

```text
QC_DICOM_Series_Dos_Femures.csv
QC_Cortes_Dos_Femures.csv
QC_Cortes_Medios_Ajustados_Dos_Femures.csv
```

Señales buenas:

```text
has_two_femurs = True
ajustado_a_corte_cercano_con_dos_femures
ajustado_por_rescate_global_a_corte_con_dos_femures
```

Señal de alerta:

```text
no_hay_cortes_con_dos_femures
```

En ese caso se debe revisar la serie DICOM seleccionada o el campo de visión.
