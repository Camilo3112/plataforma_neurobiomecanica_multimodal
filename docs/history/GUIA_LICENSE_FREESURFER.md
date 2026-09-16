# Cómo obtener `license.txt` de FreeSurfer

## 1. Entrar al registro oficial

Abre el formulario oficial de registro de FreeSurfer:

```text
https://surfer.nmr.mgh.harvard.edu/registration.html
```

## 2. Llenar el formulario

Debes completar datos básicos como nombre, institución/correo y uso académico/investigativo.

## 3. Revisar el correo

FreeSurfer envía el archivo o el contenido de la licencia al correo registrado. Puede llegar como:

- archivo `license.txt`, o
- texto con varias líneas que debes copiar en un archivo llamado `license.txt`.

## 4. Guardarlo en el proyecto

En este proyecto guárdalo exactamente aquí:

```bash
/home/humath/Escritorio/license.txt
```

Puedes crearlo manualmente así:

```bash
nano /home/humath/Escritorio/license.txt
```

Pega el contenido recibido, guarda con:

```text
Ctrl + O
Enter
Ctrl + X
```

## 5. Probar que FreeSurfer lo reconoce

```bash
export FS_LICENSE=/home/humath/Escritorio/license.txt
recon-all -version
```

Si FreeSurfer está instalado correctamente, no debería mostrar error de licencia.

## 6. Copiarlo también al directorio de FreeSurfer, si quieres

```bash
source /usr/local/freesurfer/8.0.0/SetUpFreeSurfer.sh
sudo cp /home/humath/Escritorio/license.txt $FREESURFER_HOME/license.txt
```

Si tu versión está en otra ruta, cambia `8.0.0` por la versión instalada.
