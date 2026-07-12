'use strict';

const CONFIG = Object.freeze({
  modelUrl: './model/model.json',
  labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
  inputWidth: 64,
  inputHeight: 64,
  processingWidth: 320,
  processingHeight: 180,
  captureBoxRatio: 0.58,
  captureBoxHorizontalOffsetRatio: 0.08,
  mirrorInput: true,
  predictionIntervalMs: 700,
  smoothingWindow: 3,
  minConfidence: 0.70,
  minMargin: 0.10,
});

const el = {
  video: document.querySelector('#video'),
  captureCanvas: document.querySelector('#captureCanvas'),
  modelInputCanvas: document.querySelector('#modelInputCanvas'),
  toggleButton: document.querySelector('#toggleCameraButton'),
  errorBox: document.querySelector('#errorBox'),
  opencvStatus: document.querySelector('#opencvStatus'),
  modelStatus: document.querySelector('#modelStatus'),
  cameraStatus: document.querySelector('#cameraStatus'),
  cameraMessage: document.querySelector('#cameraMessage'),
  predictionLabel: document.querySelector('#predictionLabel'),
  predictionConfidence: document.querySelector('#predictionConfidence'),
  predictionMessage: document.querySelector('#predictionMessage'),
  scores: document.querySelector('#scores'),
};

const state = {
  cv: null,
  model: null,
  stream: null,
  timer: null,
  processing: false,
  stopped: true,
  history: [],
};

document.addEventListener('DOMContentLoaded', () => {
  el.toggleButton.addEventListener('click', toggleCamera);
  initialize().catch(showError);
});

async function initialize() {
  clearError();
  const [cv] = await Promise.all([waitForOpenCv(), loadModel()]);
  state.cv = cv;
  setStatus(el.opencvStatus, 'OpenCV listo', 'ready');
  el.predictionMessage.textContent = 'Activa la cámara para comenzar.';
}

async function loadModel() {
  try {
    try {
      await tf.setBackend('webgl');
    } catch {
      // TensorFlow.js puede usar otro backend si WebGL no está disponible.
    }

    await tf.ready();
    state.model = await tf.loadLayersModel(CONFIG.modelUrl);

    const inputShape = state.model.inputs[0]?.shape;
    const outputShape = state.model.outputs[0]?.shape;

    if (!inputShape || inputShape[1] !== 64 || inputShape[2] !== 64 || inputShape[3] !== 1) {
      throw new Error(`Entrada inesperada: ${JSON.stringify(inputShape)}.`);
    }

    if (!outputShape || outputShape[1] !== CONFIG.labels.length) {
      throw new Error(`Salida inesperada: ${JSON.stringify(outputShape)}.`);
    }

    tf.tidy(() => {
      const warmup = tf.zeros([1, 64, 64, 1]);
      state.model.predict(warmup);
    });

    setStatus(el.modelStatus, 'Modelo listo', 'ready');
  } catch (error) {
    setStatus(el.modelStatus, 'Modelo no disponible', 'error');
    throw new Error(
      'No se pudo cargar ./model/model.json. Comprueba model.json y todos los .bin. '
      + getErrorMessage(error),
    );
  }
}

async function waitForOpenCv() {
  const startedAt = Date.now();

  while (Date.now() - startedAt < 30000) {
    const candidate = window.cv;

    if (candidate) {
      const cv = candidate instanceof Promise ? await candidate : candidate;

      if (cv?.Mat && cv?.imread && cv?.cvtColor && cv?.resize && cv?.imshow) {
        return cv;
      }
    }

    await sleep(50);
  }

  setStatus(el.opencvStatus, 'OpenCV no disponible', 'error');
  throw new Error('OpenCV.js no terminó de cargar. Revisa la conexión a internet.');
}

async function toggleCamera() {
  if (state.stream) {
    stopCamera();
  } else {
    await startCamera();
  }
}

async function startCamera() {
  if (!state.model || !state.cv) {
    showError(new Error('Espera a que OpenCV y el modelo terminen de cargar.'));
    return;
  }

  try {
    clearError();
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: 'user',
        width: { ideal: 640 },
        height: { ideal: 360 },
        frameRate: { ideal: 15, max: 20 },
      },
    });

    el.video.srcObject = state.stream;
    await el.video.play();
    await waitForVideoDimensions();

    setStatus(el.cameraStatus, 'Cámara activa', 'ready');
    el.toggleButton.textContent = 'Detener cámara';
    el.cameraMessage.textContent = 'Predicción activa. Mantén la mano dentro del recuadro.';

    state.history = [];
    state.stopped = false;
    scheduleNextPrediction(0);
  } catch (error) {
    stopCamera();
    showError(error);
  }
}

function stopCamera() {
  state.stopped = true;
  stopTimer();
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  state.history = [];
  el.video.srcObject = null;
  el.toggleButton.textContent = 'Activar cámara';
  el.cameraMessage.textContent = 'Cámara apagada. Actívala para comenzar.';
  setStatus(el.cameraStatus, 'Cámara apagada', '');
}

function scheduleNextPrediction(delay = CONFIG.predictionIntervalMs) {
  stopTimer();

  if (state.stopped || !state.stream) {
    return;
  }

  state.timer = window.setTimeout(async () => {
    await processFrame();
    scheduleNextPrediction();
  }, delay);
}

function stopTimer() {
  if (state.timer !== null) {
    window.clearTimeout(state.timer);
    state.timer = null;
  }
}

async function processFrame() {
  if (
    state.processing
    || state.stopped
    || !state.stream
    || !state.model
    || !state.cv
    || el.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
  ) {
    return;
  }

  state.processing = true;
  const cv = state.cv;

  el.captureCanvas.width = CONFIG.processingWidth;
  el.captureCanvas.height = CONFIG.processingHeight;

  const ctx = el.captureCanvas.getContext('2d', { willReadFrequently: true });

  if (!ctx) {
    state.processing = false;
    showError(new Error('No se pudo crear el canvas de captura.'));
    return;
  }

  // Reduce el frame antes de pasarlo a OpenCV: evita congelar el navegador.
  ctx.drawImage(el.video, 0, 0, CONFIG.processingWidth, CONFIG.processingHeight);

  let source;
  let oriented;
  let cropped;
  let gray;
  let resized;
  let floatImage;
  let inputTensor;
  let outputTensor;

  try {
    source = cv.imread(el.captureCanvas);
    oriented = new cv.Mat();

    if (CONFIG.mirrorInput) {
      cv.flip(source, oriented, 1);
    } else {
      source.copyTo(oriented);
    }

    const cropSize = Math.floor(
      Math.min(oriented.cols, oriented.rows) * CONFIG.captureBoxRatio,
    );

    const startX = Math.max(
      0,
      Math.min(
        Math.floor(oriented.cols * CONFIG.captureBoxHorizontalOffsetRatio),
        oriented.cols - cropSize,
      ),
    );

    const startY = Math.floor((oriented.rows - cropSize) / 2);

    cropped = oriented.roi(new cv.Rect(startX, startY, cropSize, cropSize));

    gray = new cv.Mat();
    cv.cvtColor(cropped, gray, cv.COLOR_RGBA2GRAY);

    resized = new cv.Mat();
    cv.resize(
      gray,
      resized,
      new cv.Size(CONFIG.inputWidth, CONFIG.inputHeight),
      0,
      0,
      cv.INTER_AREA,
    );

    cv.imshow(el.modelInputCanvas, resized);

    floatImage = new cv.Mat();
    resized.convertTo(floatImage, cv.CV_32F, 1 / 255);

    const values = new Float32Array(floatImage.data32F);

    inputTensor = tf.tensor4d(values, [1, 64, 64, 1], 'float32');

    const rawOutput = state.model.predict(inputTensor);
    outputTensor = Array.isArray(rawOutput) ? rawOutput[0] : rawOutput;

    if (!outputTensor) {
      throw new Error('El modelo no entregó una salida válida.');
    }

    const probabilities = Array.from(await outputTensor.data());
    updatePrediction(probabilities);
    clearError();

    // Permite que el navegador pinte y responda a clics antes de continuar.
    await tf.nextFrame();
  } catch (error) {
    showError(error);
    state.stopped = true;
    stopTimer();
  } finally {
    outputTensor?.dispose();
    inputTensor?.dispose();
    floatImage?.delete();
    resized?.delete();
    gray?.delete();
    cropped?.delete();
    oriented?.delete();
    source?.delete();
    state.processing = false;
  }
}

function updatePrediction(probabilities) {
  state.history.push(probabilities);

  if (state.history.length > CONFIG.smoothingWindow) {
    state.history.shift();
  }

  const averaged = CONFIG.labels.map((label, index) => {
    const confidence = state.history.reduce(
      (sum, vector) => sum + (vector[index] ?? 0),
      0,
    ) / state.history.length;

    return { label, confidence };
  }).sort((a, b) => b.confidence - a.confidence);

  const best = averaged[0];
  const second = averaged[1];

  if (!best) {
    return;
  }

  const margin = best.confidence - (second?.confidence ?? 0);
  const confident = best.confidence >= CONFIG.minConfidence && margin >= CONFIG.minMargin;

  el.predictionLabel.textContent = confident ? best.label : '?';
  el.predictionConfidence.textContent = formatConfidence(best.confidence);
  el.predictionMessage.textContent = confident
    ? `Predicción estable. Margen: ${formatConfidence(margin)}.`
    : 'Seña no reconocida con suficiente seguridad.';

  el.scores.replaceChildren(
    ...averaged.map((score) => {
      const row = document.createElement('div');
      row.className = 'score-row';

      const label = document.createElement('span');
      label.textContent = score.label;

      const confidence = document.createElement('span');
      confidence.textContent = formatConfidence(score.confidence);

      row.append(label, confidence);
      return row;
    }),
  );
}

function setStatus(element, text, statusClass) {
  element.textContent = text;
  element.classList.remove('ready', 'error');

  if (statusClass) {
    element.classList.add(statusClass);
  }
}

function showError(error) {
  el.errorBox.textContent = getErrorMessage(error);
  el.errorBox.classList.remove('hidden');
  console.error(error);
}

function clearError() {
  el.errorBox.textContent = '';
  el.errorBox.classList.add('hidden');
}

function formatConfidence(value) {
  return `${(Math.max(0, Math.min(value, 1)) * 100).toFixed(1)}%`;
}

function getErrorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function sleep(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function waitForVideoDimensions() {
  const startedAt = Date.now();

  while (Date.now() - startedAt < 10000) {
    if (el.video.videoWidth > 0 && el.video.videoHeight > 0) {
      return;
    }

    await sleep(50);
  }

  throw new Error('La cámara no entregó dimensiones válidas.');
}

window.addEventListener('beforeunload', () => {
  stopCamera();
  state.model?.dispose();
});
