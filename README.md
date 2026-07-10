# Taller 3 — Sign Language Digits

Proyecto de clasificación de dígitos en lenguaje de señas mediante una red neuronal convolucional (CNN).

## Organización

```text
sign-language-digits-taller3/
├── notebooks/
│   ├── 00_trabajo_original_companeros.ipynb
│   └── 06_evaluacion_regularizacion.ipynb
├── data/
├── models/
├── results/
│   ├── figures/
│   └── metrics/
├── app/
├── docs/
├── requirements.txt
└── .gitignore
```

## Archivos de trabajo en Google Drive

Para evitar subir archivos binarios grandes al repositorio, se recomienda guardar los datos y modelos en:

```text
Mi unidad/Taller3_Sign_Language/
├── data/
│   ├── X.npy
│   └── Y.npy
├── models/
│   └── modelo_preliminar_companeros.keras
├── logs/
└── outputs/
```

El archivo `modelo_preliminar_companeros.keras` corresponde al primer modelo entrenado. No debe llamarse “definitivo” hasta completar la comparación del punto 6.

## Notebooks

- `00_trabajo_original_companeros.ipynb`: respaldo del trabajo previo. No modificarlo.
- `06_evaluacion_regularizacion.ipynb`: notebook principal para evaluar, regularizar y comparar modelos.

Los notebooks `imdb.ipynb`, `imdb2.ipynb`, `imdb3.ipynb` e `imdb4.ipynb` son material de referencia del profesor y no forman parte del proyecto final.
