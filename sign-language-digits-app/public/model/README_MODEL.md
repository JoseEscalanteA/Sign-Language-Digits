# Modelo TensorFlow.js

Esta carpeta contiene el modelo final convertido a TensorFlow.js.

Estructura actual:

```text
public/model/
├── README_MODEL.md
├── model.json
├── group1-shard1of3.bin
├── group1-shard2of3.bin
└── group1-shard3of3.bin
```

El nombre y cantidad de archivos `.bin` puede variar si el modelo se vuelve a convertir.

## Ruta usada por la aplicacion

Angular sirve el contenido de `public/` desde la raiz del sitio. Por eso el archivo:

```text
public/model/model.json
```

se carga en el navegador desde:

```text
/model/model.json
```

## Conversion realizada

Modelo de origen:

```text
../Sign-Language-Digits/modelo_senas_definitivo.keras
```

Comando usado para convertir un modelo Keras v3:

```bash
tensorflowjs_converter --input_format=keras_keras --output_format=tfjs_layers_model modelo_senas_definitivo.keras public/model/
```

Despues de la conversion se postproceso `model.json` para que TensorFlow.js Layers pudiera cargar la topologia generada desde Keras 3.

Ajustes aplicados:

- `batch_shape` se cambio a `batch_input_shape`.
- `DTypePolicy` se simplifico a `float32`.
- Se quitaron pesos del optimizador porque no se usan para inferencia.
- Se renombraron los pesos de capas para coincidir con los nombres esperados por TensorFlow.js.

## Parametros que deben coincidir

Si el modelo se vuelve a convertir, revisar `src/app/shared/app-config.ts` y confirmar:

- `modelUrl`
- `inputWidth`
- `inputHeight`
- `inputChannels`
- `normalizeInput`
- `normalizationDivisor`
- `classLabels`

El modelo final debe haber sido entrenado con el dataset Sign Language Digits.

Para el modelo `modelo_senas_definitivo.keras`, la configuracion esperada es:

```text
inputWidth: 64
inputHeight: 64
inputChannels: 1
normalizacion: pixel / 255
tensor final: [1, 64, 64, 1]
clases: ['0','1','2','3','4','5','6','7','8','9']
```
