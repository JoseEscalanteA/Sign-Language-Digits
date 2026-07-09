# Sign Language Digits - App simple

Aplicación web sencilla para el Taller 3 de Inteligencia Artificial.

Permite usar la cámara del navegador para practicar dígitos en lenguaje de señas. La app usa:

- TensorFlow.js para cargar el modelo entrenado convertido a formato web.
- OpenCV.js para procesar la imagen de la cámara.
- HTML, CSS y JavaScript puro para evitar una estructura innecesariamente compleja.

## Estructura

```txt
sign-language-digits-simple-app/
├── index.html
├── style.css
├── app.js
├── README.md
└── model/
    ├── model.json
    └── group1-shard1of1.bin
```

## Modelo

El modelo fue entrenado en Keras y convertido a TensorFlow.js. La app carga:

```js
tf.loadLayersModel('./model/model.json')
```

Entrada esperada por el modelo:

```txt
[1, 64, 64, 1]
```

Esto significa:

- 1 imagen.
- 64 píxeles de alto.
- 64 píxeles de ancho.
- 1 canal, es decir, escala de grises.

Salida del modelo:

```txt
10 clases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

## Preprocesamiento

Antes de predecir, la app realiza estos pasos:

1. Captura un frame desde la cámara.
2. Recorta una región central donde se ubica la mano.
3. Convierte la imagen a escala de grises usando OpenCV.js.
4. Redimensiona la imagen a 64x64.
5. Normaliza los píxeles dividiendo por 255.
6. Crea un tensor con forma `[1, 64, 64, 1]`.
7. Envía el tensor al modelo para obtener la predicción.

## Cómo ejecutar

No abras `index.html` directamente con doble clic, porque el navegador puede bloquear la carga del modelo y la cámara.

Ejecuta un servidor local dentro de esta carpeta.

Con Python:

```bash
python -m http.server 5500
```

Luego abre en el navegador:

```txt
http://localhost:5500
```

También puedes usar la extensión Live Server de Visual Studio Code.

## Uso

1. Abre la aplicación desde `localhost`.
2. Presiona **Iniciar cámara**.
3. Presiona **Cargar modelo**.
4. Coloca la mano dentro del recuadro central.
5. Presiona **Predecir**.
6. Para practicar, presiona **Nuevo número** y luego **Predecir y comprobar**.

## Nota sobre compatibilidad del modelo

El archivo `model.json` fue generado desde el modelo Keras convertido a TensorFlow.js. Para facilitar la carga en el navegador, se simplificaron metadatos de compatibilidad de Keras 3, como `DTypePolicy` a `float32` y `batch_shape` a `batch_input_shape`.

Estos cambios no modifican los pesos entrenados ni la arquitectura lógica del modelo; solo ayudan a que TensorFlow.js pueda interpretar el archivo correctamente en el navegador.

## Ética y reutilización

Esta aplicación fue desarrollada como una implementación simple para el Taller 3. Si se reutilizan ideas, código o estructura de proyectos externos, como el proyecto de Alexandre Donciu-Julin mencionado en el enunciado, se debe declarar en el informe final.
