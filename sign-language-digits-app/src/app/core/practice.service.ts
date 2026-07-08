import { Injectable } from '@angular/core';
import { APP_CONFIG } from '../shared/app-config';

export interface PracticeResult {
  targetLabel: string;
  predictedLabel: string;
  isCorrect: boolean;
  message: string;
}

@Injectable({ providedIn: 'root' })
export class PracticeService {
  createTarget(labels = APP_CONFIG.classLabels, previousTarget: string | null = null): string {
    const candidates = labels.length > 1 && previousTarget
      ? labels.filter((label) => label !== previousTarget)
      : labels;

    return candidates[Math.floor(Math.random() * candidates.length)] ?? labels[0] ?? '0';
  }

  evaluate(targetLabel: string, predictedLabel: string): PracticeResult {
    const isCorrect = targetLabel === predictedLabel;

    return {
      targetLabel,
      predictedLabel,
      isCorrect,
      message: isCorrect
        ? `Correcto: la prediccion coincide con el numero ${targetLabel}.`
        : `Intenta nuevamente: se pidio ${targetLabel}, pero el modelo predijo ${predictedLabel}.`,
    };
  }
}
