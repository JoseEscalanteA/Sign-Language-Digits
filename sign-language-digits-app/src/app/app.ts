import { Component, ElementRef, OnDestroy, ViewChild, inject } from '@angular/core';
import { CameraService } from './core/camera.service';
import {
  MISSING_MODEL_MESSAGE,
  ModelService,
  type PredictionResult,
} from './core/model.service';
import { PracticeResult, PracticeService } from './core/practice.service';
import { PreprocessService } from './core/preprocess.service';
import { APP_CONFIG } from './shared/app-config';

@Component({
  selector: 'app-root',
  imports: [],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App implements OnDestroy {
  @ViewChild('videoElement') private videoElement?: ElementRef<HTMLVideoElement>;

  private readonly cameraService = inject(CameraService);
  private readonly modelService = inject(ModelService);
  private readonly preprocessService = inject(PreprocessService);
  private readonly practiceService = inject(PracticeService);

  readonly config = APP_CONFIG;
  readonly missingModelMessage = MISSING_MODEL_MESSAGE;

  cameraReady = false;
  modelLoaded = false;
  modelLoading = false;
  predicting = false;
  errorMessage: string | null = null;
  cameraMessage = 'Camara detenida. Presiona "Iniciar camara" para comenzar.';
  modelMessage = this.missingModelMessage;
  prediction: PredictionResult | null = null;
  practiceTarget = this.practiceService.createTarget(this.config.classLabels);
  practiceResult: PracticeResult | null = null;

  get canPredict(): boolean {
    return this.cameraReady && this.modelLoaded && !this.predicting;
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
      this.cameraMessage = 'Camara activa. Ubica la mano dentro del marco central.';
    } catch (error) {
      this.cameraReady = false;
      this.errorMessage = this.getErrorMessage(error);
      this.cameraMessage = 'No fue posible iniciar la camara.';
    }
  }

  stopCamera(): void {
    this.cameraService.stop();
    this.cameraReady = false;
    this.cameraMessage = 'Camara detenida.';
  }

  async loadModel(): Promise<void> {
    try {
      this.errorMessage = null;
      this.modelLoading = true;
      this.modelMessage = `Cargando modelo desde ${this.config.modelUrl}...`;
      await this.modelService.loadModel(this.config.modelUrl);
      this.modelLoaded = true;
      this.modelMessage = 'Modelo cargado correctamente. Ya se pueden realizar predicciones.';
    } catch (error) {
      this.modelLoaded = false;
      this.modelMessage = this.missingModelMessage;
      this.errorMessage = this.getErrorMessage(error);
    } finally {
      this.modelLoading = false;
    }
  }

  async predict(): Promise<void> {
    const video = this.videoElement?.nativeElement;

    if (!video) {
      this.errorMessage = 'No se encontro el elemento de video de la aplicacion.';
      return;
    }

    if (!this.canPredict) {
      this.errorMessage = 'Para predecir se necesita la camara activa y el modelo cargado.';
      return;
    }

    try {
      this.errorMessage = null;
      this.predicting = true;
      const inputTensor = await this.preprocessService.createInputTensor(video, this.config);

      try {
        this.prediction = await this.modelService.predict(inputTensor, this.config.classLabels);
        this.practiceResult = this.practiceService.evaluate(this.practiceTarget, this.prediction.label);
      } finally {
        inputTensor.dispose();
      }
    } catch (error) {
      this.errorMessage = this.getErrorMessage(error);
    } finally {
      this.predicting = false;
    }
  }

  nextPracticeTarget(): void {
    this.practiceTarget = this.practiceService.createTarget(this.config.classLabels, this.practiceTarget);
    this.practiceResult = null;
    this.prediction = null;
    this.errorMessage = null;
  }

  formatConfidence(confidence: number): string {
    return `${(confidence * 100).toFixed(1)}%`;
  }

  confidencePercent(confidence: number): number {
    return Math.max(0, Math.min(confidence * 100, 100));
  }

  ngOnDestroy(): void {
    this.cameraService.stop();
  }

  private getErrorMessage(error: unknown): string {
    if (error instanceof Error) {
      return error.message;
    }

    return String(error);
  }
}
