export interface AppConfig {
  modelUrl: string;
  inputWidth: number;
  inputHeight: number;
  inputChannels: number;
  mirrorCameraPreview: boolean;
  mirrorModelInput: boolean;
  captureBoxRatio: number;
  captureBoxPosition: 'left' | 'center' | 'right';
  captureBoxHorizontalOffsetRatio: number;
  normalizeInput: boolean;
  normalizationDivisor: number;
  predictionIntervalMs: number;
  classLabels: string[];
}

export const APP_CONFIG: AppConfig = {
  modelUrl: '/model/model.json',
  inputWidth: 64,
  inputHeight: 64,
  inputChannels: 1,
  mirrorCameraPreview: true,
  mirrorModelInput: true,
  captureBoxRatio: 0.58,
  captureBoxPosition: 'left',
  captureBoxHorizontalOffsetRatio: 0.08,
  normalizeInput: true,
  normalizationDivisor: 255,
  predictionIntervalMs: 200,
  classLabels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
};
