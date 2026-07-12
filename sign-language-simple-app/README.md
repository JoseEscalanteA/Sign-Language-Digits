# App simple: OpenCV.js + TensorFlow.js

Esta versión elimina Angular y utiliza únicamente:

- HTML
- CSS
- JavaScript
- OpenCV.js para el preprocesamiento
- TensorFlow.js para la inferencia

## Estructura

```text
sign-language-simple-app/
├── index.html
├── styles.css
├── app.js
└── model/
    ├── model.json
    ├── group1-shard1of1.bin
    └── labels.json  (opcional)
```

Copia dentro de `model/` los archivos generados por TensorFlow.js.
El nombre del archivo `.bin` puede ser distinto o puede haber más de uno.
No los renombres, porque `model.json` contiene sus referencias exactas.

## Ejecutar

No abras `index.html` con doble clic. El modelo se carga mediante `fetch`.

Desde esta carpeta ejecuta una de estas opciones:

### Python

```bash
python -m http.server 5500
```

### Node.js

```bash
npx http-server . -p 5500
```

Después abre:

```text
http://localhost:5500
```

## Flujo técnico

```text
Cámara
  -> canvas de captura
  -> OpenCV.js: cv.imread
  -> OpenCV.js: cv.flip
  -> OpenCV.js: ROI
  -> OpenCV.js: cv.cvtColor
  -> OpenCV.js: cv.resize a 64x64
  -> normalización píxel / 255
  -> TensorFlow.js: tensor [1,64,64,1]
  -> predicción CNN
```

## Configuración importante

En `app.js`, el objeto `CONFIG` contiene:

- `captureBoxRatio: 0.58`
- `captureBoxHorizontalOffsetRatio: 0.08`
- `predictionIntervalMs: 200`
- `smoothingWindow: 5`
- `minConfidence: 0.70`
- `minMargin: 0.10`

El procesamiento mantiene el mismo formato del entrenamiento:

- escala de grises
- 64 × 64 × 1
- `float32`
- rango 0 a 1

## OpenCV local para una entrega sin internet

La versión incluida carga OpenCV desde:

```html
https://docs.opencv.org/4.x/opencv.js
```

Para volverla independiente de internet:

1. Descarga `opencv.js`.
2. Guárdalo en `opencv/opencv.js`.
3. Cambia en `index.html` la URL remota por:

```html
<script async src="opencv/opencv.js"></script>
```
