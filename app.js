// Taller 3 IA - Sign Language Digits
// App simple con cámara, OpenCV.js y TensorFlow.js.
// El modelo espera tensores con forma [1, 64, 64, 1].

const MODEL_PATH = './model/model.json';
const INPUT_WIDTH = 64;
const INPUT_HEIGHT = 64;
const CLASS_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

let model = null;
let cameraReady = false;
let opencvReady = false;
let targetDigit = null;

const els = {
  status: document.getElementById('status'),
  opencvBadge: document.getElementById('opencvBadge'),
  modelBadge: document.getElementById('modelBadge'),
  cameraBadge: document.getElementById('cameraBadge'),
  video: document.getElementById('video'),
  captureCanvas: document.getElementById('captureCanvas'),
  processedCanvas: document.getElementById('processedCanvas'),
  startCameraBtn: document.getElementById('startCameraBtn'),
  loadModelBtn: document.getElementById('loadModelBtn'),
  predictBtn: document.getElementById('predictBtn'),
  prediction: document.getElementById('prediction'),
  confidence: document.getElementById('confidence'),
  newTargetBtn: document.getElementById('newTargetBtn'),
  checkBtn: document.getElementById('checkBtn'),
  targetDigit: document.getElementById('targetDigit'),
  practiceResult: document.getElementById('practiceResult'),
};

function setStatus(message) {
  els.status.textContent = message;
}

function setBadge(element, text, state = 'warning') {
  element.textContent = text;
  element.className = `badge ${state}`;
}

function refreshButtons() {
  const canPredict = cameraReady && opencvReady && Boolean(model);
  els.predictBtn.disabled = !canPredict;
  els.checkBtn.disabled = !canPredict || targetDigit === null;
}

async function waitForOpenCV() {
  setStatus('Esperando OpenCV.js...');

  return new Promise((resolve, reject) => {
    const timeoutMs = 20000;
    const startedAt = Date.now();

    const timer = setInterval(() => {
      if (window.cv && cv.Mat) {
        clearInterval(timer);
        opencvReady = true;
        setBadge(els.opencvBadge, 'OpenCV listo', 'ok');
        setStatus('OpenCV.js cargado. Puedes iniciar la cámara y cargar el modelo.');
        refreshButtons();
        resolve();
      }

      if (Date.now() - startedAt > timeoutMs) {
        clearInterval(timer);
        setBadge(els.opencvBadge, 'OpenCV no cargó', 'error');
        reject(new Error('OpenCV.js no cargó dentro del tiempo esperado.'));
      }
    }, 250);
  });
}

async function startCamera() {
  try {
    setStatus('Solicitando permiso de cámara...');

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'user',
        width: { ideal: 960 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    els.video.srcObject = stream;

    await new Promise((resolve) => {
      els.video.onloadedmetadata = () => {
        els.video.play();
        resolve();
      };
    });

    cameraReady = true;
    setBadge(els.cameraBadge, 'Cámara lista', 'ok');
    setStatus('Cámara iniciada. Ubica la mano dentro del recuadro.');
    refreshButtons();
  } catch (error) {
    console.error(error);
    setBadge(els.cameraBadge, 'Error de cámara', 'error');
    setStatus('No se pudo iniciar la cámara. Revisa permisos del navegador.');
  }
}

async function loadModel() {
  try {
    setStatus('Cargando modelo TensorFlow.js...');
    model = await tf.loadLayersModel(MODEL_PATH);

    // Calentamiento simple del modelo para confirmar que acepta [1,64,64,1].
    const warmup = tf.zeros([1, INPUT_HEIGHT, INPUT_WIDTH, 1]);
    const output = model.predict(warmup);
    await output.data();
    warmup.dispose();
    output.dispose();

    setBadge(els.modelBadge, 'Modelo listo', 'ok');
    setStatus('Modelo cargado correctamente.');
    refreshButtons();
  } catch (error) {
    console.error(error);
    setBadge(els.modelBadge, 'Error de modelo', 'error');
    setStatus('No se pudo cargar el modelo. Ejecuta la app con un servidor local y revisa la consola.');
  }
}

function preprocessFrame() {
  if (!opencvReady) {
    throw new Error('OpenCV.js todavía no está listo.');
  }

  const video = els.video;
  const canvas = els.captureCanvas;
  const ctx = canvas.getContext('2d');

  const width = video.videoWidth;
  const height = video.videoHeight;

  if (!width || !height) {
    throw new Error('La cámara todavía no entregó dimensiones de video.');
  }

  canvas.width = width;
  canvas.height = height;

  // Se invierte horizontalmente para que el canvas coincida con el video mostrado como espejo.
  ctx.save();
  ctx.translate(width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, width, height);
  ctx.restore();

  const src = cv.imread(canvas);

  // ROI: recorte central que coincide aproximadamente con el recuadro visual.
  const roiX = Math.floor(width * 0.20);
  const roiY = Math.floor(height * 0.15);
  const roiW = Math.floor(width * 0.60);
  const roiH = Math.floor(height * 0.70);
  const rect = new cv.Rect(roiX, roiY, roiW, roiH);
  const roi = src.roi(rect);

  const gray = new cv.Mat();
  const resized = new cv.Mat();

  // OpenCV procesa la imagen: RGBA -> gris -> 64x64.
  cv.cvtColor(roi, gray, cv.COLOR_RGBA2GRAY);
  cv.resize(gray, resized, new cv.Size(INPUT_WIDTH, INPUT_HEIGHT), 0, 0, cv.INTER_AREA);

  // Mostrar en pantalla la imagen que realmente entra al modelo.
  cv.imshow(els.processedCanvas, resized);

  const data = new Float32Array(INPUT_WIDTH * INPUT_HEIGHT);
  for (let i = 0; i < resized.data.length; i++) {
    data[i] = resized.data[i] / 255.0;
  }

  src.delete();
  roi.delete();
  gray.delete();
  resized.delete();

  return tf.tensor4d(data, [1, INPUT_HEIGHT, INPUT_WIDTH, 1]);
}

function getTopPrediction(probabilities) {
  let bestIndex = 0;
  let bestValue = probabilities[0];

  for (let i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > bestValue) {
      bestValue = probabilities[i];
      bestIndex = i;
    }
  }

  return {
    label: CLASS_LABELS[bestIndex],
    confidence: bestValue,
  };
}

async function predictDigit() {
  if (!model) {
    throw new Error('El modelo no está cargado.');
  }

  const inputTensor = preprocessFrame();
  const outputTensor = model.predict(inputTensor);
  const probabilities = await outputTensor.data();
  const result = getTopPrediction(probabilities);

  inputTensor.dispose();
  outputTensor.dispose();

  els.prediction.textContent = result.label;
  els.confidence.textContent = `Confianza: ${(result.confidence * 100).toFixed(2)}%`;

  return result;
}

async function handlePredict() {
  try {
    setStatus('Procesando imagen y realizando predicción...');
    await predictDigit();
    setStatus('Predicción realizada.');
  } catch (error) {
    console.error(error);
    setStatus(error.message);
  }
}

function newPracticeTarget() {
  targetDigit = CLASS_LABELS[Math.floor(Math.random() * CLASS_LABELS.length)];
  els.targetDigit.textContent = targetDigit;
  els.practiceResult.textContent = 'Resultado: esperando predicción...';
  refreshButtons();
}

async function handleCheckPractice() {
  try {
    setStatus('Comprobando el signo solicitado...');
    const result = await predictDigit();

    if (result.label === targetDigit) {
      els.practiceResult.textContent = `Resultado: correcto. El modelo predijo ${result.label}.`;
      setStatus('Correcto.');
    } else {
      els.practiceResult.textContent = `Resultado: incorrecto. Se pidió ${targetDigit}, pero el modelo predijo ${result.label}.`;
      setStatus('Intenta ajustar la posición de la mano y prueba otra vez.');
    }
  } catch (error) {
    console.error(error);
    setStatus(error.message);
  }
}

function bindEvents() {
  els.startCameraBtn.addEventListener('click', startCamera);
  els.loadModelBtn.addEventListener('click', loadModel);
  els.predictBtn.addEventListener('click', handlePredict);
  els.newTargetBtn.addEventListener('click', newPracticeTarget);
  els.checkBtn.addEventListener('click', handleCheckPractice);
}

async function init() {
  bindEvents();
  refreshButtons();

  try {
    await waitForOpenCV();
  } catch (error) {
    console.error(error);
    setStatus('OpenCV.js no pudo cargarse. Revisa tu conexión a internet.');
  }
}

init();
