"use strict";

const CONFIG = Object.freeze({
  modelPath: "./model/model.json",
  labelsPath: "./model/labels.json",
  openCvPath: "https://docs.opencv.org/4.13.0/opencv.js",
  inputSize: 64,
  captureSize: 300,
  roiRatio: 0.72,
  predictionDelayMs: 250,
  minimumConfidence: 0.65,
  smoothingWindow: 5,
  stableVotes: 4,
  resourceTimeoutMs: 25000,
  labels: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
});

const elements = {
  protocolWarning: document.getElementById("protocolWarning"),
  statusText: document.getElementById("statusText"),
  tfStatus: document.getElementById("tfStatus"),
  cvStatus: document.getElementById("cvStatus"),
  modelStatus: document.getElementById("modelStatus"),
  cameraStatus: document.getElementById("cameraStatus"),
  video: document.getElementById("video"),
  cameraPlaceholder: document.getElementById("cameraPlaceholder"),
  captureCanvas: document.getElementById("captureCanvas"),
  processedCanvas: document.getElementById("processedCanvas"),
  startCameraBtn: document.getElementById("startCameraBtn"),
  switchCameraBtn: document.getElementById("switchCameraBtn"),
  stopCameraBtn: document.getElementById("stopCameraBtn"),
  reloadModelBtn: document.getElementById("reloadModelBtn"),
  prediction: document.getElementById("prediction"),
  confidence: document.getElementById("confidence"),
  stability: document.getElementById("stability"),
  topPredictions: document.getElementById("topPredictions"),
  targetDigit: document.getElementById("targetDigit"),
  practiceFeedback: document.getElementById("practiceFeedback"),
  newTargetBtn: document.getElementById("newTargetBtn"),
  challengeCount: document.getElementById("challengeCount"),
  successCount: document.getElementById("successCount"),
  streakCount: document.getElementById("streakCount")
};

const resourceElements = {
  tensorflow: elements.tfStatus,
  opencv: elements.cvStatus,
  model: elements.modelStatus,
  camera: elements.cameraStatus
};

const resourceNames = {
  tensorflow: "TensorFlow.js",
  opencv: "OpenCV.js",
  model: "modelo",
  camera: "cámara"
};

const state = {
  resources: {
    tensorflow: { status: "pending", detail: "" },
    opencv: { status: "pending", detail: "" },
    model: { status: "pending", detail: "" },
    camera: { status: "pending", detail: "" }
  },
  cv: null,
  model: null,
  labels: [],
  stream: null,
  facingMode: "user",
  cameraRequest: 0,
  predictionTimer: null,
  predictionRunning: false,
  probabilityHistory: [],
  predictionError: "",
  target: null,
  challengeFailed: false,
  challengeCount: 0,
  successCount: 0,
  streakCount: 0
};

function setResource(name, status, detail = "") {
  state.resources[name] = { status, detail };

  const badge = resourceElements[name];
  badge.dataset.state = status;
  badge.title = detail || `${resourceNames[name]}: ${status}`;
  badge.setAttribute(
    "aria-label",
    `${resourceNames[name]}: ${status}${detail ? `. ${detail}` : ""}`
  );

  renderOverallStatus();
}

function renderOverallStatus() {
  const entries = Object.entries(state.resources);
  const errors = entries.filter(([, resource]) => resource.status === "error");

  if (errors.length > 0) {
    elements.statusText.textContent = errors
      .map(([name, resource]) => `${resourceNames[name]}: ${resource.detail}`)
      .join(" · ");
    return;
  }

  if (state.predictionError) {
    elements.statusText.textContent = state.predictionError;
    return;
  }

  const loading = entries
    .filter(([, resource]) => resource.status === "loading")
    .map(([name]) => resourceNames[name]);

  if (loading.length > 0) {
    elements.statusText.textContent = `Cargando ${loading.join(", ")}…`;
    return;
  }

  const inferenceReady = ["tensorflow", "opencv", "model"].every(
    (name) => state.resources[name].status === "ready"
  );

  if (inferenceReady && state.resources.camera.status === "ready") {
    elements.statusText.textContent = "Predicción automática activa.";
  } else if (inferenceReady) {
    elements.statusText.textContent = "Recursos listos. Inicia la cámara para predecir.";
  } else {
    elements.statusText.textContent = "Preparando recursos de inferencia…";
  }
}

function withTimeout(promise, timeoutMs, message) {
  let timeoutId;

  const timeout = new Promise((_, reject) => {
    timeoutId = window.setTimeout(() => reject(new Error(message)), timeoutMs);
  });

  return Promise.race([promise, timeout]).finally(() => {
    window.clearTimeout(timeoutId);
  });
}

async function initializeTensorFlow() {
  setResource("tensorflow", "loading", "Inicializando el backend");

  try {
    if (!window.tf) {
      throw new Error("la biblioteca no se descargó; revisa la conexión a internet");
    }

    await withTimeout(
      window.tf.ready(),
      CONFIG.resourceTimeoutMs,
      "se agotó el tiempo de inicialización"
    );

    setResource("tensorflow", "ready", `Backend: ${window.tf.getBackend()}`);
  } catch (error) {
    setResource("tensorflow", "error", getErrorMessage(error));
    throw error;
  }
}

function waitForOpenCvRuntime() {
  return new Promise((resolve, reject) => {
    const finish = async () => {
      try {
        let cvInstance = window.cv;

        if (!cvInstance) {
          throw new Error("el script se descargó, pero no creó el objeto cv");
        }

        if (typeof cvInstance.then === "function") {
          cvInstance = await cvInstance;
          window.cv = cvInstance;
        }

        if (cvInstance.Mat) {
          resolve(cvInstance);
          return;
        }

        const previousCallback = cvInstance.onRuntimeInitialized;
        cvInstance.onRuntimeInitialized = () => {
          if (typeof previousCallback === "function") {
            previousCallback();
          }
          resolve(cvInstance);
        };
      } catch (error) {
        reject(error);
      }
    };

    if (window.cv) {
      finish();
      return;
    }

    const script = document.createElement("script");
    script.src = CONFIG.openCvPath;
    script.async = true;
    script.id = "opencvScript";
    script.onload = finish;
    script.onerror = () => reject(new Error("no se pudo descargar el script"));
    document.head.appendChild(script);
  });
}

async function initializeOpenCv() {
  setResource("opencv", "loading", "Descargando y preparando el runtime");

  try {
    state.cv = await withTimeout(
      waitForOpenCvRuntime(),
      CONFIG.resourceTimeoutMs,
      "se agotó el tiempo de carga; revisa la conexión a internet"
    );
    setResource("opencv", "ready", "Runtime inicializado");
  } catch (error) {
    setResource("opencv", "error", getErrorMessage(error));
    throw error;
  }
}

async function loadModel() {
  if (state.resources.tensorflow.status !== "ready") {
    setResource("model", "error", "TensorFlow.js todavía no está disponible");
    return;
  }

  setResource("model", "loading", "Cargando arquitectura, pesos y etiquetas");
  elements.reloadModelBtn.disabled = true;

  let loadedModel = null;

  try {
    const labelsResponse = await fetch(CONFIG.labelsPath, { cache: "no-store" });
    if (!labelsResponse.ok) {
      throw new Error(`labels.json respondió HTTP ${labelsResponse.status}`);
    }

    const loadedLabels = (await labelsResponse.json()).map(String);
    if (!arraysEqual(loadedLabels, CONFIG.labels)) {
      throw new Error("labels.json debe contener exactamente los dígitos del 0 al 9");
    }

    loadedModel = await window.tf.loadLayersModel(CONFIG.modelPath);
    validateModelShape(loadedModel, loadedLabels.length);

    const warmupInput = window.tf.zeros([
      1,
      CONFIG.inputSize,
      CONFIG.inputSize,
      1
    ]);
    const warmupOutput = loadedModel.predict(warmupInput);
    const warmupTensors = Array.isArray(warmupOutput) ? warmupOutput : [warmupOutput];

    try {
      await warmupTensors[0].data();
    } finally {
      warmupInput.dispose();
      warmupTensors.forEach((tensor) => tensor.dispose());
    }

    if (state.model) {
      state.model.dispose();
    }

    state.model = loadedModel;
    state.labels = loadedLabels;
    loadedModel = null;
    resetPredictionHistory();

    setResource("model", "ready", "Entrada 64 × 64 × 1; salida de 10 clases");
    startPredictionLoop();
  } catch (error) {
    if (loadedModel) {
      loadedModel.dispose();
    }
    setResource("model", "error", getErrorMessage(error));
  } finally {
    elements.reloadModelBtn.disabled = false;
  }
}

function validateModelShape(model, numberOfLabels) {
  const inputShape = model.inputs[0]?.shape;
  const outputShape = model.outputs[0]?.shape;
  const expectedInput = [null, CONFIG.inputSize, CONFIG.inputSize, 1];

  if (!inputShape || !arraysEqual(inputShape, expectedInput)) {
    throw new Error(`forma de entrada incompatible: ${String(inputShape)}`);
  }

  if (!outputShape || outputShape.at(-1) !== numberOfLabels) {
    throw new Error(`forma de salida incompatible: ${String(outputShape)}`);
  }
}

async function startCamera() {
  const requestId = ++state.cameraRequest;
  stopCamera(false);
  setResource("camera", "loading", "Solicitando permiso al navegador");
  elements.startCameraBtn.disabled = true;

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("el navegador no admite acceso a cámara o el contexto no es seguro");
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: state.facingMode },
        width: { ideal: 960 },
        height: { ideal: 720 }
      }
    });

    if (requestId !== state.cameraRequest) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }

    state.stream = stream;
    elements.video.srcObject = stream;
    await waitForVideoMetadata();
    await elements.video.play();

    elements.video.classList.toggle("mirrored", state.facingMode === "user");
    elements.cameraPlaceholder.hidden = true;
    elements.switchCameraBtn.disabled = false;
    elements.stopCameraBtn.disabled = false;

    setResource("camera", "ready", `Orientación: ${state.facingMode}`);
    resetPredictionHistory();
    startPredictionLoop();
  } catch (error) {
    if (requestId !== state.cameraRequest) {
      return;
    }

    stopCamera(false);
    setResource("camera", "error", cameraErrorMessage(error));
  } finally {
    if (requestId === state.cameraRequest) {
      elements.startCameraBtn.disabled = false;
    }
  }
}

function waitForVideoMetadata() {
  if (elements.video.readyState >= 1 && elements.video.videoWidth > 0) {
    return Promise.resolve();
  }

  return withTimeout(
    new Promise((resolve) => {
      elements.video.addEventListener("loadedmetadata", resolve, { once: true });
    }),
    10000,
    "la cámara no entregó información de video"
  );
}

function stopCamera(updateResource = true) {
  stopPredictionLoop();

  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }

  elements.video.srcObject = null;
  elements.video.classList.remove("mirrored");
  elements.cameraPlaceholder.hidden = false;
  elements.switchCameraBtn.disabled = true;
  elements.stopCameraBtn.disabled = true;
  resetPredictionHistory();

  if (updateResource) {
    setResource("camera", "pending", "Cámara detenida");
  }
}

async function switchCamera() {
  state.facingMode = state.facingMode === "user" ? "environment" : "user";
  await startCamera();
}

function inferenceIsReady() {
  return ["tensorflow", "opencv", "model", "camera"].every(
    (name) => state.resources[name].status === "ready"
  );
}

function startPredictionLoop() {
  if (!inferenceIsReady() || state.predictionTimer !== null) {
    return;
  }

  schedulePrediction(0);
}

function schedulePrediction(delay) {
  window.clearTimeout(state.predictionTimer);
  state.predictionTimer = window.setTimeout(runPredictionCycle, delay);
}

function stopPredictionLoop() {
  if (state.predictionTimer !== null) {
    window.clearTimeout(state.predictionTimer);
    state.predictionTimer = null;
  }
}

async function runPredictionCycle() {
  state.predictionTimer = null;

  if (!inferenceIsReady() || state.predictionRunning) {
    return;
  }

  state.predictionRunning = true;

  try {
    const probabilities = await predictCurrentFrame();
    state.predictionError = "";
    updateSmoothedPrediction(probabilities);
    renderOverallStatus();
  } catch (error) {
    state.predictionError = `Predicción: ${getErrorMessage(error)}`;
    renderOverallStatus();
  } finally {
    state.predictionRunning = false;
    if (inferenceIsReady()) {
      schedulePrediction(CONFIG.predictionDelayMs);
    }
  }
}

async function predictCurrentFrame() {
  let inputTensor = null;
  let outputTensors = [];

  try {
    inputTensor = preprocessCurrentFrame();
    const prediction = state.model.predict(inputTensor);
    outputTensors = Array.isArray(prediction) ? prediction : [prediction];

    if (!outputTensors[0]) {
      throw new Error("el modelo no produjo una salida");
    }

    return Array.from(await outputTensors[0].data());
  } finally {
    if (inputTensor) {
      inputTensor.dispose();
    }
    outputTensors.forEach((tensor) => tensor.dispose());
  }
}

function preprocessCurrentFrame() {
  if (elements.video.readyState < 2 || !elements.video.videoWidth) {
    throw new Error("la cámara todavía no tiene un frame disponible");
  }

  const videoWidth = elements.video.videoWidth;
  const videoHeight = elements.video.videoHeight;
  const cropSize = Math.floor(Math.min(videoWidth, videoHeight) * CONFIG.roiRatio);
  const cropX = Math.floor((videoWidth - cropSize) / 2);
  const cropY = Math.floor((videoHeight - cropSize) / 2);
  const context = elements.captureCanvas.getContext("2d", { willReadFrequently: true });

  context.drawImage(
    elements.video,
    cropX,
    cropY,
    cropSize,
    cropSize,
    0,
    0,
    CONFIG.captureSize,
    CONFIG.captureSize
  );

  let source = null;
  let gray = null;
  let resized = null;

  try {
    source = state.cv.imread(elements.captureCanvas);
    gray = new state.cv.Mat();
    resized = new state.cv.Mat();

    state.cv.cvtColor(source, gray, state.cv.COLOR_RGBA2GRAY);
    state.cv.resize(
      gray,
      resized,
      new state.cv.Size(CONFIG.inputSize, CONFIG.inputSize),
      0,
      0,
      state.cv.INTER_AREA
    );

    // La normalización directa coincide con data/X.npy. No se invierte el gris.
    state.cv.imshow(elements.processedCanvas, resized);

    const normalizedPixels = new Float32Array(CONFIG.inputSize * CONFIG.inputSize);
    for (let index = 0; index < resized.data.length; index += 1) {
      normalizedPixels[index] = resized.data[index] / 255;
    }

    return window.tf.tensor4d(
      normalizedPixels,
      [1, CONFIG.inputSize, CONFIG.inputSize, 1]
    );
  } finally {
    if (resized) resized.delete();
    if (gray) gray.delete();
    if (source) source.delete();
  }
}

function updateSmoothedPrediction(probabilities) {
  if (probabilities.length !== state.labels.length) {
    throw new Error("la salida no coincide con la cantidad de etiquetas");
  }

  state.probabilityHistory.push(probabilities);
  if (state.probabilityHistory.length > CONFIG.smoothingWindow) {
    state.probabilityHistory.shift();
  }

  const averages = probabilities.map((_, index) => {
    const total = state.probabilityHistory.reduce(
      (sum, sample) => sum + sample[index],
      0
    );
    return total / state.probabilityHistory.length;
  });

  const bestIndex = indexOfMaximum(averages);
  const votes = state.probabilityHistory.filter(
    (sample) => indexOfMaximum(sample) === bestIndex
  ).length;
  const enoughSamples = state.probabilityHistory.length === CONFIG.smoothingWindow;
  const stable = enoughSamples && votes >= CONFIG.stableVotes;
  const confident = averages[bestIndex] >= CONFIG.minimumConfidence;
  const result = {
    label: state.labels[bestIndex],
    confidence: averages[bestIndex],
    stable,
    confident,
    votes,
    sampleCount: state.probabilityHistory.length
  };

  renderPrediction(result, averages);
  evaluatePractice(result);
}

function renderPrediction(result, averages) {
  const recognized = result.stable && result.confident;
  elements.prediction.textContent = recognized ? result.label : "?";
  elements.confidence.textContent = `Confianza: ${(result.confidence * 100).toFixed(1)} %`;

  if (state.probabilityHistory.length < CONFIG.smoothingWindow) {
    elements.stability.textContent = `Calibrando ${result.sampleCount}/${CONFIG.smoothingWindow}`;
  } else if (!result.stable) {
    elements.stability.textContent = "Mantén la mano quieta";
  } else if (!result.confident) {
    elements.stability.textContent = "Confianza insuficiente";
  } else {
    elements.stability.textContent = `Predicción estable (${result.votes}/${CONFIG.smoothingWindow})`;
  }

  const topThree = averages
    .map((confidence, index) => ({ label: state.labels[index], confidence }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 3);

  elements.topPredictions.replaceChildren(
    ...topThree.map((score) => {
      const row = document.createElement("div");
      row.className = "score-row";

      const label = document.createElement("strong");
      label.textContent = `Dígito ${score.label}`;

      const track = document.createElement("div");
      track.className = "score-track";
      const fill = document.createElement("div");
      fill.className = "score-fill";
      fill.style.width = `${Math.max(0, Math.min(100, score.confidence * 100))}%`;
      track.appendChild(fill);

      const value = document.createElement("span");
      value.className = "score-value";
      value.textContent = `${(score.confidence * 100).toFixed(1)} %`;

      row.append(label, track, value);
      return row;
    })
  );
}

function resetPredictionHistory() {
  state.probabilityHistory = [];
  state.predictionError = "";
  elements.prediction.textContent = "—";
  elements.confidence.textContent = "Confianza: —";
  elements.stability.textContent = state.stream ? "Calibrando" : "Esperando cámara";
  elements.topPredictions.innerHTML = '<p class="empty-state">Aún no hay predicciones.</p>';
}

function createPracticeTarget() {
  if (state.target !== null) {
    state.streakCount = 0;
  }

  let nextTarget;
  do {
    nextTarget = CONFIG.labels[Math.floor(Math.random() * CONFIG.labels.length)];
  } while (CONFIG.labels.length > 1 && nextTarget === state.target);

  state.target = nextTarget;
  state.challengeFailed = false;
  state.challengeCount += 1;
  elements.targetDigit.textContent = nextTarget;
  elements.practiceFeedback.classList.remove("success");
  elements.practiceFeedback.textContent = inferenceIsReady()
    ? "Realiza el signo y mantenlo estable dentro del recuadro."
    : "Desafío preparado. Inicia la cámara y espera que los recursos estén listos.";
  renderPracticeStats();
}

function evaluatePractice(result) {
  if (state.target === null || !result.stable || !result.confident) {
    return;
  }

  if (result.label === state.target) {
    state.successCount += 1;
    state.streakCount += 1;
    state.target = null;
    state.challengeFailed = false;
    elements.practiceFeedback.classList.add("success");
    elements.practiceFeedback.textContent =
      `Correcto. El modelo reconoció el dígito ${result.label}. Genera otro desafío.`;
    renderPracticeStats();
    return;
  }

  if (!state.challengeFailed) {
    state.streakCount = 0;
    state.challengeFailed = true;
    renderPracticeStats();
  }

  elements.practiceFeedback.classList.remove("success");
  elements.practiceFeedback.textContent =
    `Se reconoce ${result.label}, pero se solicita ${state.target}. Ajusta la mano e inténtalo otra vez.`;
}

function renderPracticeStats() {
  elements.challengeCount.textContent = String(state.challengeCount);
  elements.successCount.textContent = String(state.successCount);
  elements.streakCount.textContent = String(state.streakCount);
}

function indexOfMaximum(values) {
  let bestIndex = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > values[bestIndex]) {
      bestIndex = index;
    }
  }
  return bestIndex;
}

function arraysEqual(first, second) {
  return first.length === second.length && first.every(
    (value, index) => value === second[index]
  );
}

function getErrorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function cameraErrorMessage(error) {
  if (error?.name === "NotAllowedError") {
    return "permiso denegado; habilita la cámara en el navegador";
  }
  if (error?.name === "NotFoundError") {
    return "no se encontró una cámara disponible";
  }
  if (error?.name === "NotReadableError") {
    return "la cámara está siendo utilizada por otra aplicación";
  }
  return getErrorMessage(error);
}

function bindEvents() {
  elements.startCameraBtn.addEventListener("click", startCamera);
  elements.switchCameraBtn.addEventListener("click", switchCamera);
  elements.stopCameraBtn.addEventListener("click", () => stopCamera(true));
  elements.reloadModelBtn.addEventListener("click", loadModel);
  elements.newTargetBtn.addEventListener("click", createPracticeTarget);

  window.addEventListener("beforeunload", () => {
    stopCamera(false);
    if (state.model) {
      state.model.dispose();
      state.model = null;
    }
  });
}

async function initializeApplication() {
  bindEvents();
  renderPracticeStats();

  if (window.location.protocol === "file:") {
    elements.protocolWarning.hidden = false;
    elements.startCameraBtn.disabled = true;
    elements.reloadModelBtn.disabled = true;
    setResource("tensorflow", "error", "la página fue abierta con file://");
    setResource("opencv", "error", "la página fue abierta con file://");
    setResource("model", "error", "usa un servidor local para cargar los archivos");
    setResource("camera", "error", "usa http://localhost para solicitar permisos");
    return;
  }

  const tensorflowAndModel = initializeTensorFlow().then(loadModel);
  await Promise.allSettled([tensorflowAndModel, initializeOpenCv()]);
  startPredictionLoop();
}

initializeApplication();
