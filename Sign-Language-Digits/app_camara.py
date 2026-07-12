import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

# 1. Cargar tu modelo entrenado
print("[DEBUG] 1. Iniciando carga del modelo...")
model = load_model('modelov2.keras') 
print("[DEBUG] 2. ¡Modelo cargado con éxito en la memoria!")

class_labels = {i: str(i) for i in range(10)}

# ========================================================
# EL FIX PARA WINDOWS: Forzar DirectShow (cv2.CAP_DSHOW)
# ========================================================
print("[DEBUG] 3. Intentando conectar con la cámara web...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# ========================================================

print("[DEBUG] 4. Configurando resolución...")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("[DEBUG] 5. ¡Cámara lista para transmitir!")

fondo_guardado = None
calibrado = False
x, y, w, h = 400, 200, 300, 300

# --- Suavizado temporal de predicciones ---
# Promediamos las probabilidades de los últimos N frames en vez de confiar
# en un solo frame (que es ruidoso por naturaleza de la sustracción de fondo).
N_FRAMES_SUAVIZADO = 8
buffer_predicciones = deque(maxlen=N_FRAMES_SUAVIZADO)

texto = "Presiona 'C' con la pared vacia para calibrar"
color_texto = (0, 165, 255)

print("\n--- INSTRUCCIONES ---")
print("1. Sal de la vista de la cámara.")
print("2. Presiona 'C' para calibrar el fondo.")
print("3. Entra al cuadro y pon tu mano dentro del cuadrado.")
print("4. Presiona 'Q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] No se pudo leer el cuadro de la cámara. ¿Está desconectada o en uso?")
        break

    frame = cv2.flip(frame, 1)
    roi = frame[y:y+h, x:x+w].copy()
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('c'):
        fondo_guardado = roi_gray.copy()
        calibrado = True
        print("\n¡Fondo calibrado!")

    if calibrado and fondo_guardado is not None:
        # 1. Sustracción de fondo y limpieza
        diff = cv2.absdiff(roi_gray, fondo_guardado)
        _, mascara = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        # Cierre morfológico: une huecos entre dedos / zonas con sombra que
        # el threshold deja separadas, para que la mano quede como un solo
        # blob sólido en vez de varios contornos chicos.
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)
        mascara = cv2.erode(mascara, kernel, iterations=1)
        mascara = cv2.dilate(mascara, kernel, iterations=1)

        mano_limpia = cv2.bitwise_and(roi_gray, roi_gray, mask=mascara)

        # 2. Auto-zoom: recortamos y centramos la mano dentro de un cuadro
        # negro, para que ocupe una proporción estable del frame en vez de
        # depender de qué tan cerca de la cámara esté tu mano.
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # En vez de quedarnos solo con el contorno más grande (que puede
        # dejar fuera dedos que quedaron como blobs separados), unimos el
        # bounding box de todos los contornos "significativos".
        contornos_validos = [c for c in contornos if cv2.contourArea(c) > 300]

        if contornos_validos:
            area_total = sum(cv2.contourArea(c) for c in contornos_validos)

            if area_total > 2000:
                xc = min(cv2.boundingRect(c)[0] for c in contornos_validos)
                yc = min(cv2.boundingRect(c)[1] for c in contornos_validos)
                xc2 = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contornos_validos)
                yc2 = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contornos_validos)
                wc, hc = xc2 - xc, yc2 - yc

                # Margen alrededor de la mano: sin esto el recorte queda
                # pegado al contorno y la mano llena todo el cuadro (mucho
                # zoom, distinto a como se ve en el dataset de entrenamiento).
                margen = int(max(wc, hc) * 0.35)
                xc = max(0, xc - margen)
                yc = max(0, yc - margen)
                wc = min(mano_limpia.shape[1] - xc, wc + 2 * margen)
                hc = min(mano_limpia.shape[0] - yc, hc + 2 * margen)

                mano_recortada = mano_limpia[yc:yc+hc, xc:xc+wc]

                dimension_max = max(wc, hc)
                cuadro_perfecto = np.zeros((dimension_max, dimension_max), dtype=np.uint8)

                offset_x = (dimension_max - wc) // 2
                offset_y = (dimension_max - hc) // 2
                cuadro_perfecto[offset_y:offset_y+hc, offset_x:offset_x+wc] = mano_recortada

                roi_resized = cv2.resize(cuadro_perfecto, (64, 64))
                # El modelo espera píxeles crudos 0-255 (así se entrenó, sin /255)
                roi_expanded = np.expand_dims(roi_resized.astype('float32'), axis=(0, -1))

                predicciones = model.predict(roi_expanded, verbose=0)[0]
                buffer_predicciones.append(predicciones)

                predicciones_suavizadas = np.mean(buffer_predicciones, axis=0)
                numero_ganador = np.argmax(predicciones_suavizadas)
                confianza = predicciones_suavizadas[numero_ganador] * 100

                if confianza > 80 and len(buffer_predicciones) == N_FRAMES_SUAVIZADO:
                    texto = f"Numero: {numero_ganador} (Confianza: {confianza:.1f}%)"
                    color_texto = (0, 255, 0)
                else:
                    texto = "Pensando..."
                    color_texto = (0, 0, 255)

                mano_limpia_bgr = cv2.cvtColor(cv2.resize(cuadro_perfecto, (150, 150)), cv2.COLOR_GRAY2BGR)
                frame[10:160, 10:160] = mano_limpia_bgr

                cv2.rectangle(frame, (x+xc, y+yc), (x+xc+wc, y+yc+hc), (255, 0, 0), 2)
            else:
                buffer_predicciones.clear()
                texto = "Pensando..."
                color_texto = (0, 0, 255)
        else:
            buffer_predicciones.clear()
            texto = "Pensando..."
            color_texto = (0, 0, 255)
    else:
        texto = "Presiona 'C' con la pared vacia para calibrar"
        color_texto = (0, 165, 255)
        buffer_predicciones.clear()

    cv2.rectangle(frame, (x, y), (x+w, y+h), color_texto, 2)
    cv2.putText(frame, texto, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_texto, 2, cv2.LINE_AA)
    cv2.imshow("Traductor de Lenguaje de Senas - Taller IA", frame)

cap.release()
cv2.destroyAllWindows()