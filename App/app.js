const MODEL_URL = "modelos/tfjs_model/model.json?v=2026071202";
const LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
const MIRROR_MODEL_INPUT = true;
const PREDICTION_INTERVAL_MS = 200;
const SMOOTHING_WINDOW = 3;

let model = null;
let stream = null;
let loopId = null;
let openCvReady = false;
let preprocessingMode = "normal";
let probabilityHistory = [];

const statusEl = document.getElementById("status");
const modelStatusEl = document.getElementById("modelStatus");
const cvStatusEl = document.getElementById("cvStatus");
const cameraStatusEl = document.getElementById("cameraStatus");
const cameraBtn = document.getElementById("cameraBtn");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const video = document.getElementById("video");
const roiBox = document.getElementById("roiBox");
const hiddenCanvas = document.getElementById("hiddenCanvas");
const roiCanvas = document.getElementById("roiCanvas");
const predictionEl = document.getElementById("prediction");
const confidenceEl = document.getElementById("confidence");
const scoresGrid = document.getElementById("scoresGrid");
const normalModeBtn = document.getElementById("normalModeBtn");
const contrastModeBtn = document.getElementById("contrastModeBtn");
const modeInfoEl = document.getElementById("modeInfo");

function setStatus(message) {
  statusEl.textContent = message;
  console.log("[APP]", message);
}

function setChip(element, text, state = "") {
  element.textContent = text;
  element.classList.remove("ready", "error");
  if (state) element.classList.add(state);
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitForOpenCV(timeoutMs = 30000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (window.cv && typeof cv.Mat === "function" && typeof cv.imread === "function") {
      return true;
    }
    await sleep(100);
  }
  return false;
}

async function loadDependencies() {
  cameraBtn.disabled = true;
  try {
    setStatus("Preparando OpenCV y el modelo...");
    const [cvLoaded, loadedModel] = await Promise.all([
      waitForOpenCV(),
      tf.loadLayersModel(MODEL_URL)
    ]);
    if (!cvLoaded) throw new Error("OpenCV.js no termino de cargar.");

    openCvReady = true;
    model = loadedModel;
    const dummy = tf.zeros([1, 64, 64, 1]);
    const warmup = model.predict(dummy);
    warmup.dispose();
    dummy.dispose();

    setChip(modelStatusEl, "Modelo listo", "ready");
    setChip(cvStatusEl, "OpenCV listo", "ready");
    setStatus("Todo listo. Activa la camara para comenzar.");
    cameraBtn.disabled = false;
  } catch (error) {
    console.error(error);
    setChip(modelStatusEl, model ? "Modelo listo" : "Modelo con error", model ? "ready" : "error");
    setChip(cvStatusEl, openCvReady ? "OpenCV listo" : "OpenCV con error", openCvReady ? "ready" : "error");
    setStatus("Error de carga: " + error.message);
  }
}

async function toggleCamera() {
  if (stream) {
    stopCamera();
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
    probabilityHistory = [];
    cameraPlaceholder.classList.add("hidden-state");
    cameraBtn.textContent = "Detener camara";
    setChip(cameraStatusEl, "Camara activa", "ready");
    setStatus("Prediccion automatica activa. Ubica la mano izquierda en el recuadro.");
    schedulePrediction(0);
  } catch (error) {
    stream = null;
    setChip(cameraStatusEl, "Camara con error", "error");
    setStatus("No fue posible iniciar la camara: " + error.message);
  }
}

function stopCamera() {
  if (loopId !== null) window.clearTimeout(loopId);
  loopId = null;
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  probabilityHistory = [];
  cameraPlaceholder.classList.remove("hidden-state");
  cameraBtn.textContent = "Activar camara";
  setChip(cameraStatusEl, "Camara inactiva");
  setStatus("Camara detenida. La ultima prediccion queda visible.");
}

function setPreprocessingMode(mode) {
  preprocessingMode = mode;
  probabilityHistory = [];
  normalModeBtn.classList.toggle("active", mode === "normal");
  contrastModeBtn.classList.toggle("active", mode === "contrast");
  modeInfoEl.textContent = mode === "normal"
    ? "Entrada igual al entrenamiento: gris / 255."
    : "Contraste local experimental antes de / 255.";
}

function getVisibleRoiInVideoPixels() {
  const width = video.videoWidth;
  const height = video.videoHeight;
  const videoRect = video.getBoundingClientRect();
  const boxRect = roiBox.getBoundingClientRect();
  const scaleX = width / videoRect.width;
  const scaleY = height / videoRect.height;
  const visualX = (boxRect.left - videoRect.left) * scaleX;
  const roiWidth = boxRect.width * scaleX;
  const roiHeight = boxRect.height * scaleY;

  // Compensa el espejo visual para recortar la region fisica correcta.
  return {
    x: Math.max(0, Math.round(width - visualX - roiWidth)),
    y: Math.max(0, Math.round((boxRect.top - videoRect.top) * scaleY)),
    width: Math.max(1, Math.round(roiWidth)),
    height: Math.max(1, Math.round(roiHeight))
  };
}

function createInputTensor() {
  if (!stream || !openCvReady || !video.videoWidth || !video.videoHeight) return null;
  hiddenCanvas.width = video.videoWidth;
  hiddenCanvas.height = video.videoHeight;
  const context = hiddenCanvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(video, 0, 0, hiddenCanvas.width, hiddenCanvas.height);

  let source = null;
  let cropped = null;
  let oriented = null;
  let gray = null;
  let resized = null;
  let processed = null;

  try {
    source = cv.imread(hiddenCanvas);
    const roi = getVisibleRoiInVideoPixels();
    const width = Math.min(roi.width, source.cols - roi.x);
    const height = Math.min(roi.height, source.rows - roi.y);
    cropped = source.roi(new cv.Rect(roi.x, roi.y, width, height));
    oriented = new cv.Mat();
    gray = new cv.Mat();
    resized = new cv.Mat();
    processed = new cv.Mat();

    if (MIRROR_MODEL_INPUT) {
      cv.flip(cropped, oriented, 1);
    } else {
      cropped.copyTo(oriented);
    }

    // El canvas y el tensor nacen de la misma matriz orientada. De este modo,
    // la miniatura 64x64 coincide horizontalmente con la vista de la camara.
    cv.cvtColor(oriented, gray, cv.COLOR_RGBA2GRAY);
    cv.resize(gray, resized, new cv.Size(64, 64), 0, 0, cv.INTER_AREA);
    if (preprocessingMode === "contrast") {
      cv.normalize(resized, processed, 0, 255, cv.NORM_MINMAX);
    } else {
      resized.copyTo(processed);
    }

    cv.imshow(roiCanvas, processed);
    const values = Float32Array.from(processed.data, value => value / 255);
    return tf.tensor4d(values, [1, 64, 64, 1]);
  } finally {
    [processed, resized, gray, oriented, cropped, source].forEach(mat => mat?.delete());
  }
}

function smoothProbabilities(probabilities) {
  probabilityHistory.push(Array.from(probabilities));
  if (probabilityHistory.length > SMOOTHING_WINDOW) probabilityHistory.shift();
  const weights = probabilityHistory.map((_, index) => 2 ** index);
  const total = weights.reduce((sum, value) => sum + value, 0);
  return LABELS.map((_, classIndex) =>
    probabilityHistory.reduce(
      (sum, frame, frameIndex) => sum + frame[classIndex] * weights[frameIndex],
      0
    ) / total
  );
}

function renderPrediction(probabilities) {
  let bestIndex = 0;
  probabilities.forEach((value, index) => {
    if (value > probabilities[bestIndex]) bestIndex = index;
  });
  predictionEl.textContent = LABELS[bestIndex];
  confidenceEl.textContent = (probabilities[bestIndex] * 100).toFixed(1) + "%";

  scoresGrid.innerHTML = "";
  probabilities.forEach((probability, index) => {
    const row = document.createElement("div");
    row.className = "score-row";
    row.innerHTML = `
      <div class="score-label"><span>${LABELS[index]}</span><span>${(probability * 100).toFixed(1)}%</span></div>
      <div class="score-track"><div class="score-fill" style="width:${Math.min(100, probability * 100)}%"></div></div>
    `;
    scoresGrid.appendChild(row);
  });
}

function schedulePrediction(delay = PREDICTION_INTERVAL_MS) {
  if (!stream) return;
  loopId = window.setTimeout(runPrediction, delay);
}

async function runPrediction() {
  if (!stream || !model) return;
  let input = null;
  let output = null;
  try {
    input = createInputTensor();
    if (!input) return;
    output = model.predict(input);
    const probabilities = smoothProbabilities(await output.data());
    renderPrediction(probabilities);
  } catch (error) {
    console.error("Error de prediccion:", error);
    setStatus("Error al predecir: " + error.message);
  } finally {
    output?.dispose();
    input?.dispose();
    schedulePrediction();
  }
}

cameraBtn.addEventListener("click", toggleCamera);
normalModeBtn.addEventListener("click", () => setPreprocessingMode("normal"));
contrastModeBtn.addEventListener("click", () => setPreprocessingMode("contrast"));
window.addEventListener("load", loadDependencies);
window.addEventListener("beforeunload", stopCamera);
