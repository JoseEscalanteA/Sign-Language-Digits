# AGENTS.md

## Objetivo del proyecto

Este proyecto corresponde al punto 7 del Taller 3 de Inteligencia Artificial de la Universidad de Tarapaca: **Redes Neuronales Convolucionales (CNN): Sign Language Digits**.

El objetivo es construir una aplicacion web que permita reconocer digitos en lenguaje de señas usando la camara del navegador y un modelo entrenado con el dataset Sign Language Digits.

La aplicacion carga el modelo convertido a TensorFlow.js desde:

```text
/model/model.json
```

## Alcance actual

El modelo final ya fue convertido a TensorFlow.js. La aplicacion debe:

- Permitir iniciar la camara del navegador.
- Mostrar la camara como elemento principal y un recuadro derecho para ubicar la mano.
- Cargar automaticamente el modelo TensorFlow.js al iniciar la aplicacion.
- Preparar la imagen con tamaño y normalizacion parametrizados.
- Ejecutar predicciones automaticas mientras la camara este activa.
- Mostrar la prediccion real solo cuando exista un modelo cargado y una imagen de camara disponible.

No se deben simular predicciones como si fueran resultados reales del modelo.

## Tecnologia usada

- Angular standalone, sin SSR.
- Tailwind CSS v3 para estilos.
- TensorFlow.js para cargar y ejecutar el modelo en el navegador.
- API `navigator.mediaDevices.getUserMedia` para usar la camara.

## Organizacion del codigo

```text
src/app/shared/app-config.ts
```

Contiene parametros editables del proyecto: ruta del modelo, tamaño de entrada, normalizacion y orden de clases.

```text
src/app/core/camera.service.ts
```

Controla el acceso a la camara.

```text
src/app/core/model.service.ts
```

Carga el modelo TensorFlow.js y ejecuta predicciones.

```text
src/app/core/preprocess.service.ts
```

Transforma la imagen del video antes de enviarla al modelo.

```text
public/model/
```

Carpeta reservada para `model.json` y los archivos `.bin` generados por TensorFlow.js.

## Restricciones importantes

- El modelo final debe entrenarse con Sign Language Digits, no con Rock Paper Scissors.
- El proyecto de Alexandre Donciu-Julin se puede usar como referencia metodologica, pero no como modelo final.
- No activar SSR, porque la camara y TensorFlow.js dependen del navegador.
- Mantener el codigo separado por responsabilidad.
- Evitar acoplar valores fijos de entrada directamente en los servicios. Usar `APP_CONFIG`.
- No agregar archivos del modelo al azar. Solo copiar el modelo final convertido cuando el equipo lo entregue.
- El modelo actual espera tensores con forma `[1, 64, 64, 1]`: imagen 64x64 en escala de grises normalizada dividiendo por 255.
- El recorte de preprocesamiento debe coincidir con el recuadro derecho mostrado sobre la camara.

## Pendiente del equipo

- Probar la aplicacion con camara real en distintas condiciones de iluminacion.
- Ajustar `captureBoxRatio` si el marco visual no coincide bien con la mano.
- Validar que el orden de clases del modelo siga coincidiendo con `APP_CONFIG.classLabels` si se reentrena o reconvierte.
- Documentar en el informe final la conversion desde `.keras` a TensorFlow.js y el uso de IA generativa si corresponde.

## Etica y atribucion

Toda seccion de codigo reutilizada desde proyectos externos, tutoriales, videos o herramientas de IA generativa debe declararse en el informe final.

El proyecto de Alexandre Donciu-Julin sobre Rock Paper Scissors CNN se considera una referencia tecnica y metodologica. Si se reutiliza codigo de ese proyecto o de cualquier otra fuente, debe indicarse claramente.
