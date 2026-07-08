# Sign Language Digits App

Aplicacion web del punto 7 del Taller 3 de Inteligencia Artificial de la Universidad de Tarapaca: **Redes Neuronales Convolucionales (CNN): Sign Language Digits**.

La aplicacion permite reconocer digitos en lenguaje de señas usando la camara del navegador y un modelo CNN convertido a TensorFlow.js.

## Estado actual

El modelo `modelo_senas_definitivo.keras` ya fue convertido a TensorFlow.js y conectado a la app. La aplicacion:

- Carga automaticamente el modelo al abrir la aplicacion.
- Inicia la camara del navegador cuando el usuario concede permiso.
- Muestra la camara como elemento principal y un recuadro derecho para ubicar la mano.
- Carga el modelo TensorFlow.js desde `/model/model.json`.
- Preprocesa la imagen antes de enviarla al modelo como tensor `[1, 64, 64, 1]`.
- Ejecuta predicciones automaticas cada `200 ms` mientras la camara esta activa.
- No simula predicciones si el modelo no esta disponible.
- Muestra `Modelo no encontrado. Debe convertirse el archivo .keras a TensorFlow.js` si falta `model.json`.

## Requisitos

- Node.js
- npm
- Angular CLI

Versiones usadas durante la creacion inicial:

```text
Angular CLI 20.1.6
Node 24.12.0
npm 11.6.2
```

## Instalacion

```bash
npm install
```

## Ejecucion en desarrollo

```bash
npm start
```

Luego abrir:

```text
http://localhost:4200/
```

El navegador pedira permiso para usar la camara.

## Modelo TensorFlow.js

La aplicacion encuentra el modelo convertido en:

```text
public/model/model.json
```

En ejecucion, Angular lo sirve como:

```text
/model/model.json
```

Archivos actuales del modelo:

```text
public/model/model.json
public/model/group1-shard1of3.bin
public/model/group1-shard2of3.bin
public/model/group1-shard3of3.bin
```

El modelo fue convertido desde:

```text
../Sign-Language-Digits/modelo_senas_definitivo.keras
```

Comando de conversion usado para Keras v3:

```bash
tensorflowjs_converter --input_format=keras_keras --output_format=tfjs_layers_model modelo_senas_definitivo.keras public/model/
```

Despues de convertir, se ajusto `model.json` para compatibilidad con TensorFlow.js Layers: `batch_shape` se adapto a `batch_input_shape`, `DTypePolicy` se simplifico a `float32`, se quitaron pesos del optimizador y se conservaron solo los pesos de inferencia.

## Parametros editables

La configuracion principal esta en:

```text
src/app/shared/app-config.ts
```

Parametros importantes:

- `modelUrl`: ruta del modelo TensorFlow.js.
- `inputWidth`: ancho esperado por el modelo.
- `inputHeight`: alto esperado por el modelo.
- `inputChannels`: cantidad de canales esperada por el modelo.
- `mirrorCameraPreview`: voltea horizontalmente la camara y el tensor de entrada para que la vista sea natural para el usuario.
- `captureBoxRatio`: tamaño relativo del recuadro usado para capturar la mano.
- `captureBoxPosition`: posicion del recorte usado por el modelo. Actualmente es `right`.
- `captureBoxHorizontalOffsetRatio`: separacion horizontal del recuadro respecto del borde derecho.
- `normalizeInput`: activa o desactiva la normalizacion.
- `normalizationDivisor`: divisor usado para normalizar pixeles, por ejemplo `255`.
- `predictionIntervalMs`: intervalo entre predicciones automaticas.
- `classLabels`: orden de clases esperado por el modelo.

Para el modelo `modelo_senas_definitivo.keras`, la app queda configurada con:

```text
entrada: 64x64
canales: 1
color: escala de grises
normalizacion: pixeles / 255
tensor final: [1, 64, 64, 1]
clases: ['0','1','2','3','4','5','6','7','8','9']
```

## Uso

1. Abrir la aplicacion y esperar que el modelo cargue automaticamente.
2. Presionar `Activar camara`.
3. Ubicar la mano dentro del recuadro derecho.
4. Observar el digito predicho y el porcentaje de confianza.
5. Presionar `Detener camara` para finalizar la captura.

## Estructura relevante

```text
src/app/core/camera.service.ts       # Camara del navegador
src/app/core/model.service.ts        # Carga y prediccion TensorFlow.js
src/app/core/preprocess.service.ts   # Recorte, resize y normalizacion
src/app/shared/app-config.ts         # Parametros del modelo
public/model/README_MODEL.md         # Detalle del modelo convertido
```

## Nota sobre el modelo final

El modelo final debe entrenarse con el dataset Sign Language Digits. El proyecto Rock Paper Scissors CNN de Alexandre Donciu-Julin se usa solo como referencia tecnica y metodologica.

## Etica y atribucion

Si se reutiliza codigo del proyecto de Alexandre Donciu-Julin, de otros autores, de tutoriales o generado mediante IA, debe declararse explicitamente en el informe final del taller.

Esta aplicacion fue estructurada para el taller y deja separadas las partes propias del proyecto, la carga del modelo y los parametros de preprocesamiento usados antes de la inferencia.
