const MODEL_URL = "modelos/tfjs_model/model.json?v=2026070202";
const LABELS = ["9", "0", "7", "6", "1", "8", "4", "3", "2", "5"];
// Orden de etiquetas de Y.npy usado por el modelo del notebook oficial.


// El modelo fue entrenado con imágenes donde la mano queda clara
// y el fondo predominantemente oscuro.
// La inversión se mantiene desactivada porque esta configuración
// obtuvo mejores resultados en las pruebas realizadas con la cámara.
// Cambiar a true únicamente si la captura muestra fondo claro y mano oscura.
const INVERT_PIXELS = false;

let model = null;
let stream = null;
let loopId = null;
let openCvReady = false;

const statusEl = document.getElementById("status");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const video = document.getElementById("video");
const hiddenCanvas = document.getElementById("hiddenCanvas");
const roiCanvas = document.getElementById("roiCanvas");
const predictionEl = document.getElementById("prediction");
const confidenceEl = document.getElementById("confidence");
const topPredictionsEl = document.getElementById("topPredictions");

function setStatus(message) {
  statusEl.textContent = message;
  console.log("[APP]", message);
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitForOpenCV(timeoutMs = 15000) {
  const start = Date.now();

  while (Date.now() - start < timeoutMs) {
    if (window.cv && typeof cv.imread === "function" && typeof cv.Mat === "function") {
      openCvReady = true;
      return true;
    }
    await sleep(150);
  }

  openCvReady = false;
  return false;
}

async function loadModelAndLibraries() {
  try {
    startBtn.disabled = true;
    setStatus("Cargando OpenCV.js...");

    const cvOk = await waitForOpenCV();
    if (!cvOk) {
      throw new Error("OpenCV.js no terminó de cargar. Revisa la conexión a internet.");
    }

    setStatus("Cargando modelo TensorFlow.js...");
    model = await tf.loadLayersModel(MODEL_URL);

    // Warm-up para evitar que la primera predicción sea lenta.
    const dummy = tf.zeros([1, 64, 64, 1]);
    model.predict(dummy).dispose();
    dummy.dispose();

    setStatus("Modelo y OpenCV.js cargados. Puedes iniciar la cámara.");
    startBtn.disabled = false;
  } catch (error) {
    console.error("Error al cargar modelo o librerías:", error);
    setStatus("Error al cargar modelo o librerías: " + error.message);
    startBtn.disabled = true;
  }
}

async function startCamera() {
  try {
    if (!model || !openCvReady) {
      setStatus("Aún no se cargan completamente el modelo y OpenCV.js.");
      return;
    }

    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    startBtn.disabled = true;
    stopBtn.disabled = false;
    setStatus("Cámara activa. Realizando predicciones...");

    predictLoop();
  } catch (error) {
    console.error("Error al iniciar cámara:", error);
    setStatus("Error al iniciar cámara: " + error.message);
  }
}

function stopCamera() {
  if (loopId) cancelAnimationFrame(loopId);
  loopId = null;

  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }

  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("Cámara detenida. Modelo cargado.");
}

function preprocessFrameWithOpenCV() {
  const width = video.videoWidth;
  const height = video.videoHeight;

  if (!width || !height || !openCvReady) return null;

  hiddenCanvas.width = width;
  hiddenCanvas.height = height;

  const ctx = hiddenCanvas.getContext("2d");
  ctx.drawImage(video, 0, 0, width, height);

  let src = null;
  let roi = null;
  let gray = null;
  let resized = null;
  let processed = null;

  try {
    src = cv.imread(hiddenCanvas); // RGBA desde el canvas.

    const roiSize = Math.floor(Math.min(width, height) * 0.45);
    const x = Math.floor((width - roiSize) / 2);
    const y = Math.floor((height - roiSize) / 2);
    const rect = new cv.Rect(x, y, roiSize, roiSize);

    roi = src.roi(rect);
    cv.flip(roi, roi, 1);
    gray = new cv.Mat();
    resized = new cv.Mat();
    processed = new cv.Mat();

    // Preprocesamiento real con OpenCV.js.
    cv.cvtColor(roi, gray, cv.COLOR_RGBA2GRAY);
    cv.resize(gray, resized, new cv.Size(64, 64), 0, 0, cv.INTER_AREA);

    if (INVERT_PIXELS) {
      cv.bitwise_not(resized, processed);
    } else {
      processed = resized.clone();
    }

    cv.imshow(roiCanvas, processed);

    const values = new Float32Array(64 * 64);
    for (let i = 0; i < processed.data.length; i++) {
      values[i] = processed.data[i] / 255.0;
    }

    return tf.tensor4d(values, [1, 64, 64, 1]);
  } catch (error) {
    console.error("Error en preprocesamiento OpenCV.js:", error);
    return null;
  } finally {
    if (src) src.delete();
    if (roi) roi.delete();
    if (gray) gray.delete();
    if (resized) resized.delete();
    if (processed) processed.delete();
  }
}

async function predictLoop() {
  if (!stream) return;

  const input = preprocessFrameWithOpenCV();

  if (input && model) {
    const prediction = model.predict(input);
    const probs = await prediction.data();

    let maxIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[maxIdx]) maxIdx = i;
    }

    const top = Array.from(probs)
      .map((prob, i) => ({ label: LABELS[i], prob }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 3);

    predictionEl.textContent = LABELS[maxIdx];
    confidenceEl.textContent = "Confianza: " + (probs[maxIdx] * 100).toFixed(1) + "%";

    topPredictionsEl.innerHTML = "";
    top.forEach(item => {
      const li = document.createElement("li");
      li.textContent = `${item.label}: ${(item.prob * 100).toFixed(1)}%`;
      topPredictionsEl.appendChild(li);
    });

    prediction.dispose();
  }

  if (input) input.dispose();

  loopId = requestAnimationFrame(predictLoop);
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
window.addEventListener("load", loadModelAndLibraries);
