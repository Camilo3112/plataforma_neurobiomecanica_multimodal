# Carpeta de referencias IEEE y plantilla LaTeX

Esta carpeta contiene una plantilla mínima para documentar la plataforma neurobiomecánica multimodal en formato IEEE.

## Archivos

- `main.tex`: documento base en LaTeX con estructura IEEE y citas BibTeX.
- `referencias.bib`: bibliografía BibTeX consolidada.
- `referencias.md`: referencias convertidas a estilo IEEE para revisión rápida en Markdown.

## Compilación sugerida

En Overleaf, sube los tres archivos y compila `main.tex`.

En local:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
