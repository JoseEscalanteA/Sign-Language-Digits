# Sign Language Digits App

Aplicacion web simple con interfaz inmersiva, OpenCV.js y TensorFlow.js.

## Ejecutar

```bash
cd App
python -m http.server 8000
```

Abrir `http://localhost:8000`.

## Entrada del modelo

El modo inicial `Normal` replica el entrenamiento final:

- Recorte coincidente con el recuadro visible.
- Entrada espejada para coincidir visualmente con la camara y las imagenes de entrenamiento.
- Escala de grises.
- Redimensionamiento a `64x64`.
- Division de pixeles por `255`.
- Tensor `[1,64,64,1]`.
- Clases `[0,1,2,3,4,5,6,7,8,9]`.

`Contraste` aplica normalizacion min-max con OpenCV como opcion experimental. No se aplica segmentacion de piel obligatoria porque puede eliminar partes de los dedos bajo iluminacion variable.
