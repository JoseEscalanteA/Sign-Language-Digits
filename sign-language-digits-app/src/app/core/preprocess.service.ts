import { Injectable } from '@angular/core';
import type { Tensor3D, Tensor4D } from '@tensorflow/tfjs';
import { APP_CONFIG, type AppConfig } from '../shared/app-config';

type TensorFlowJs = typeof import('@tensorflow/tfjs');

@Injectable({ providedIn: 'root' })
export class PreprocessService {
  private tfPromise: Promise<TensorFlowJs> | null = null;

  async createInputTensor(
    videoElement: HTMLVideoElement,
    config: AppConfig = APP_CONFIG,
  ): Promise<Tensor4D> {
    if (videoElement.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      throw new Error('La camara aun no tiene una imagen disponible.');
    }

    const tf = await this.loadTensorFlow();

    // Se recorta el centro porque coincide con el marco visual donde el usuario ubica la mano.
    return tf.tidy(() => {
      const image = tf.browser.fromPixels(videoElement, 3);
      const [height, width] = image.shape;
      const captureRatio = Math.min(Math.max(config.captureBoxRatio, 0.2), 1);
      const cropSize = Math.floor(Math.min(width, height) * captureRatio);
      const startX = Math.floor((width - cropSize) / 2);
      const startY = Math.floor((height - cropSize) / 2);
      const cropped = image.slice([startY, startX, 0], [cropSize, cropSize, 3]);
      const resized = tf.image
        .resizeBilinear(cropped, [config.inputHeight, config.inputWidth], true)
        .toFloat() as Tensor3D;

      const modelImage = this.matchModelChannels(resized, config.inputChannels);
      const normalized = config.normalizeInput
        ? modelImage.div(config.normalizationDivisor)
        : modelImage;

      return normalized.expandDims(0) as Tensor4D;
    });
  }

  private matchModelChannels(image: Tensor3D, inputChannels: number): Tensor3D {
    if (inputChannels === 3) {
      return image;
    }

    if (inputChannels !== 1) {
      throw new Error(`Cantidad de canales no soportada: ${inputChannels}. Use 1 o 3.`);
    }

    // El modelo final fue entrenado con imagenes en escala de grises: [64, 64, 1].
    const [red, green, blue] = image.split(3, 2) as [Tensor3D, Tensor3D, Tensor3D];

    return red.mul(0.299).add(green.mul(0.587)).add(blue.mul(0.114)) as Tensor3D;
  }

  private async loadTensorFlow(): Promise<TensorFlowJs> {
    this.tfPromise ??= import('@tensorflow/tfjs');
    const tf = await this.tfPromise;
    await tf.ready();

    return tf;
  }
}
