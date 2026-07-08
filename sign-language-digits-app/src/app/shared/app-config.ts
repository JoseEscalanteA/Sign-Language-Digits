export interface AppConfig {
  modelUrl: string;
  inputWidth: number;
  inputHeight: number;
  inputChannels: number;
  captureBoxRatio: number;
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
  captureBoxRatio: 0.72,
  normalizeInput: true,
  normalizationDivisor: 255,
  predictionIntervalMs: 500,
  classLabels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
};
