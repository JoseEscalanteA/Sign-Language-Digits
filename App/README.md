# Aplicación Sign Language Digits

Aplicación web para probar el modelo CNN entrenado con Sign Language Digits. Usa TensorFlow.js para cargar el modelo exportado y OpenCV.js para procesar la imagen de la cámara.

## Archivos principales

- `index.html`: interfaz web.
- `styles.css`: estilos visuales.
- `app.js`: carga del modelo, cámara, preprocesamiento con OpenCV.js y predicción.
- `modelos/tfjs_model/model.json`: estructura del modelo para TensorFlow.js.
- `modelos/tfjs_model/group1-shard*.bin`: pesos del modelo.

## Cómo ejecutar

La cámara no funciona correctamente abriendo el archivo con doble clic. Debe ejecutarse con servidor local:

```bash
cd app
python -m http.server 8000
```

Luego abrir:

```text
http://localhost:8000
```

## Preprocesamiento usado

1. Captura del frame desde la cámara.
2. Recorte de región central.
3. Conversión a escala de grises con `cv.cvtColor`.
4. Redimensionamiento a `64x64` con `cv.resize`.
5. Visualización con `cv.imshow`.
6. Normalización a `[0,1]`.
7. Predicción con TensorFlow.js.

## Orden de etiquetas

El modelo incluido usa el orden:

```text
["9", "0", "7", "6", "1", "8", "4", "3", "2", "5"]
```

Ese orden coincide con `Y.npy` usado por el notebook original. No cambiarlo en `app.js` salvo que el modelo se reentrene con otro orden de etiquetas.
