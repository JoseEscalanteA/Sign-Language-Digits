import { Injectable } from '@angular/core';
import type { LayersModel, Tensor } from '@tensorflow/tfjs';
import { APP_CONFIG } from '../shared/app-config';

type TensorFlowJs = typeof import('@tensorflow/tfjs');

export interface PredictionScore {
  label: string;
  confidence: number;
}

export interface PredictionResult {
  label: string;
  confidence: number;
  scores: PredictionScore[];
}

export const MISSING_MODEL_MESSAGE =
  'Modelo no encontrado. Debe convertirse el archivo .keras a TensorFlow.js';

@Injectable({ providedIn: 'root' })
export class ModelService {
  private tf: TensorFlowJs | null = null;
  private model: LayersModel | null = null;

  get isLoaded(): boolean {
    return this.model !== null;
  }

  async loadModel(modelUrl = APP_CONFIG.modelUrl): Promise<void> {
    try {
      this.tf = await import('@tensorflow/tfjs');
      await this.tf.ready();
      this.model = await this.tf.loadLayersModel(modelUrl);
    } catch (error) {
      this.model = null;
      throw new Error(
        `${MISSING_MODEL_MESSAGE}. Ruta esperada: ${modelUrl}. Detalle: ${this.getErrorMessage(error)}`,
      );
    }
  }

  async predict(inputTensor: Tensor, classLabels = APP_CONFIG.classLabels): Promise<PredictionResult> {
    if (!this.model) {
      throw new Error('El modelo aun no esta cargado.');
    }

    const rawPrediction = this.model.predict(inputTensor);
    const predictionTensors = Array.isArray(rawPrediction) ? rawPrediction : [rawPrediction];
    const outputTensor = predictionTensors[0];

    if (!outputTensor) {
      throw new Error('El modelo no entrego una salida valida.');
    }

    try {
      const values = Array.from(await outputTensor.data());
      const scores = values
        .map((confidence, index) => ({
          label: classLabels[index] ?? String(index),
          confidence,
        }))
        .sort((a, b) => b.confidence - a.confidence);

      const bestScore = scores[0];

      if (!bestScore) {
        throw new Error('La salida del modelo esta vacia.');
      }

      return {
        label: bestScore.label,
        confidence: bestScore.confidence,
        scores,
      };
    } finally {
      predictionTensors.forEach((tensor) => tensor.dispose());
    }
  }

  private getErrorMessage(error: unknown): string {
    if (error instanceof Error) {
      return error.message;
    }

    return String(error);
  }
}
