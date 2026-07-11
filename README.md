# Sign Language Digits - aplicación web

Aplicación desarrollada para el Taller 3 de Inteligencia Artificial de la
Universidad de Tarapacá. Utiliza una cámara, OpenCV.js y TensorFlow.js para
reconocer dígitos del 0 al 9 representados en lenguaje de señas.

## Requisitos

- Navegador actualizado con acceso a cámara.
- Conexión a internet para descargar TensorFlow.js y OpenCV.js.
- Python 3, Live Server u otro servidor HTTP local.

La cámara funciona en `localhost` o mediante HTTPS. No se debe abrir
`index.html` directamente con doble clic porque `file://` puede bloquear el
modelo, las etiquetas y el acceso a la cámara.

## Ejecución

Desde la carpeta raíz del proyecto:

```bash
python -m http.server 5500
```

Después, abrir:

```text
http://localhost:5500
```

La interfaz muestra por separado el estado de TensorFlow.js, OpenCV.js, el
modelo y la cámara. Cuando los tres recursos de inferencia estén listos, se
puede presionar **Iniciar cámara**.

La aplicación fija TensorFlow.js 4.22.0 y OpenCV.js 4.13.0 para evitar cambios
involuntarios de comportamiento entre ejecuciones.

## Estructura

```text
.
|-- index.html
|-- style.css
|-- app.js
|-- README.md
|-- data/
|   |-- X.npy
|   `-- Y.npy
|-- model/
|   |-- model.json
|   |-- group1-shard1of1.bin
|   `-- labels.json
`-- notebooks/
    |-- Evaluacion_regularizacion.ipynb
    `-- Conversion_tensorflowjs.ipynb
```

## Contrato de entrada

El modelo desplegado espera un tensor `float32` con forma:

```text
[1, 64, 64, 1]
```

La aplicación procesa cada frame de esta manera:

1. Recorta una región de interés cuadrada y centrada.
2. Convierte la región de RGBA a escala de grises con OpenCV.js.
3. Redimensiona la imagen a 64 x 64 mediante `INTER_AREA`.
4. Normaliza cada píxel dividiéndolo por 255.
5. Construye el tensor `[1, 64, 64, 1]`.

No se invierten los valores. La normalización `gris / 255` fue comprobada
comparando `data/X.npy` con las fotografías originales del dataset.

La vista de la cámara frontal se muestra como espejo para que su uso resulte
natural, pero el tensor conserva la orientación original de la cámara. Esto es
coherente con el entrenamiento, que no utilizó `horizontal_flip`.

## Predicción y práctica

La aplicación promedia las últimas cinco salidas del modelo. Una predicción se
acepta cuando al menos cuatro coinciden y la confianza promedio es igual o
superior al 65 %. Si no se cumplen esas condiciones se muestra `?`, en vez de
presentar una clase poco confiable como definitiva.

El modo práctica genera un dígito aleatorio y confirma el acierto solamente
cuando la predicción es estable.

## Modelo

- Entrada: imágenes 64 x 64 en escala de grises.
- Salida: 10 probabilidades softmax, correspondientes a los dígitos 0-9.
- Modelo final evaluado: CNN seleccionada con regularización L1 durante el
  entrenamiento.
- Precisión registrada sobre el conjunto de prueba: 98,07 %.

El modelo de inferencia convertido a TensorFlow.js no incluye el término de
regularización L1. Esto es intencional: la regularización interviene en la
función de pérdida durante el entrenamiento, pero no modifica el cálculo de
`predict()`.

La precisión del conjunto de prueba corresponde a datos provenientes de la
misma fuente de entrenamiento. No garantiza el mismo resultado con todas las
cámaras, personas, distancias, fondos o condiciones de iluminación.

## Solución de problemas

- **La página indica `file://`:** ejecutar el servidor local descrito arriba.
- **OpenCV.js o TensorFlow.js no cargan:** revisar la conexión a internet y los
  bloqueadores de contenido.
- **La cámara es rechazada:** habilitar el permiso en la configuración del
  navegador y cerrar otras aplicaciones que la estén utilizando.
- **La confianza es baja:** centrar la mano completa, mantenerla quieta,
  mejorar la iluminación y utilizar un fondo uniforme.

## Ética, reutilización y uso de IA

- Dataset: *Sign Language Digits Dataset*, publicado por Arda Mavi y
  estudiantes de Ankara Ayrancı Anadolu High School.
- El planteamiento de aplicación con cámara y región de interés toma como
  referencia conceptual el proyecto *RockPaperScissorsCNN* de Alexandre
  Donciu-Julin, indicado en el enunciado del taller. Esta aplicación no copia
  literalmente su notebook de OpenCV.
- La revisión, depuración, estructura HTML/CSS/JavaScript, manejo de recursos,
  estabilización de predicciones y este README fueron desarrollados con apoyo
  de IA generativa mediante OpenCode (modelo de OpenAI), en julio de 2026.
- Los notebooks del proyecto contienen el entrenamiento, la evaluación y la
  conversión del modelo realizados por el equipo. Cualquier otro fragmento
  reutilizado debe identificarse adicionalmente en el informe final.
