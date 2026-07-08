import { Component, ElementRef, OnDestroy, OnInit, ViewChild, inject } from '@angular/core';
import { CameraService } from './core/camera.service';
import {
  MISSING_MODEL_MESSAGE,
  ModelService,
  type PredictionResult,
} from './core/model.service';
import { PreprocessService } from './core/preprocess.service';
import { APP_CONFIG } from './shared/app-config';

@Component({
  selector: 'app-root',
  imports: [],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App implements OnInit, OnDestroy {
  @ViewChild('videoElement') private videoElement?: ElementRef<HTMLVideoElement>;

  private readonly cameraService = inject(CameraService);
  private readonly modelService = inject(ModelService);
  private readonly preprocessService = inject(PreprocessService);
  private predictionIntervalId: number | null = null;

  readonly config = APP_CONFIG;
  readonly missingModelMessage = MISSING_MODEL_MESSAGE;

  cameraReady = false;
  modelLoaded = false;
  modelLoading = true;
  predicting = false;
  errorMessage: string | null = null;
  cameraMessage = 'Camara desactivada. Activa la camara para iniciar predicciones.';
  modelMessage = 'Cargando modelo TensorFlow.js...';
  prediction: PredictionResult | null = null;

  get automaticPredictionActive(): boolean {
    return this.cameraReady && this.modelLoaded;
  }

  ngOnInit(): void {
    void this.loadModel();
  }

  async toggleCamera(): Promise<void> {
    if (this.cameraReady) {
      this.stopCamera();
      return;
    }

    await this.startCamera();
  }

  async startCamera(): Promise<void> {
    const video = this.videoElement?.nativeElement;

    if (!video) {
      this.errorMessage = 'No se encontro el elemento de video de la aplicacion.';
      return;
    }

    try {
      this.errorMessage = null;
      await this.cameraService.start(video);
      this.cameraReady = true;
      this.cameraMessage = this.modelLoaded
        ? 'Prediccion automatica activa. Ubica la mano izquierda en el recuadro.'
        : 'Camara activa. Esperando que el modelo termine de cargar.';
      this.startPredictionLoop();
    } catch (error) {
      this.cameraReady = false;
      this.errorMessage = this.getErrorMessage(error);
      this.cameraMessage = 'No fue posible iniciar la camara.';
    }
  }

  stopCamera(): void {
    this.stopPredictionLoop();
    this.cameraService.stop();
    this.cameraReady = false;
    this.cameraMessage = 'Camara desactivada. La ultima prediccion queda visible como referencia.';
  }

  private async loadModel(): Promise<void> {
    try {
      this.errorMessage = null;
      this.modelLoading = true;
      this.modelMessage = `Cargando modelo desde ${this.config.modelUrl}...`;
      await this.modelService.loadModel(this.config.modelUrl);
      this.modelLoaded = true;
      this.modelMessage = 'Modelo cargado. Las predicciones se ejecutan automaticamente con la camara activa.';
      this.startPredictionLoop();
    } catch (error) {
      this.modelLoaded = false;
      this.modelMessage = this.missingModelMessage;
      this.errorMessage = this.getErrorMessage(error);
    } finally {
      this.modelLoading = false;
    }
  }

  private async runPredictionOnce(): Promise<void> {
    const video = this.videoElement?.nativeElement;

    if (!video || !this.cameraReady || !this.modelLoaded || this.predicting) {
      return;
    }

    try {
      this.predicting = true;
      const inputTensor = await this.preprocessService.createInputTensor(video, this.config);

      try {
        this.prediction = await this.modelService.predict(inputTensor, this.config.classLabels);
        this.errorMessage = null;
      } finally {
        inputTensor.dispose();
      }
    } catch (error) {
      const message = this.getErrorMessage(error);

      if (!message.startsWith('La camara aun no tiene')) {
        this.errorMessage = message;
      }
    } finally {
      this.predicting = false;
    }
  }

  formatConfidence(confidence: number): string {
    return `${(confidence * 100).toFixed(1)}%`;
  }

  confidencePercent(confidence: number): number {
    return Math.max(0, Math.min(confidence * 100, 100));
  }

  ngOnDestroy(): void {
    this.stopPredictionLoop();
    this.cameraService.stop();
  }

  private startPredictionLoop(): void {
    if (this.predictionIntervalId !== null || !this.cameraReady || !this.modelLoaded) {
      return;
    }

    this.cameraMessage = 'Prediccion automatica activa. Ubica la mano izquierda en el recuadro.';
    void this.runPredictionOnce();

    this.predictionIntervalId = window.setInterval(() => {
      void this.runPredictionOnce();
    }, this.config.predictionIntervalMs);
  }

  private stopPredictionLoop(): void {
    if (this.predictionIntervalId === null) {
      return;
    }

    window.clearInterval(this.predictionIntervalId);
    this.predictionIntervalId = null;
  }

  private getErrorMessage(error: unknown): string {
    if (error instanceof Error) {
      return error.message;
    }

    return String(error);
  }
}
