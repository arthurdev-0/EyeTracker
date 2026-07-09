import cv2
import numpy as np
from collections import deque
import pyautogui
import math
import threading
import time
import keyboard

MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2
mouse_control_enabled = True
filter_length = 8

# Carregar cascata de detecção de rosto (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Variáveis de rastreamento
calibration_offset_yaw = 0
calibration_offset_pitch = 0
ray_origins = deque(maxlen=filter_length)
ray_directions = deque(maxlen=filter_length)
raw_yaw_deg = 0
raw_pitch_deg = 0

# Posição alvo do mouse
mouse_target = [CENTER_X, CENTER_Y]
mouse_lock = threading.Lock()

# Abrir câmera
cap = None
for camera_idx in [0, 1]:
    temp_cap = cv2.VideoCapture(camera_idx)
    if temp_cap.isOpened():
        cap = temp_cap
        print(f"Câmera encontrada no índice {camera_idx}")
        break
    temp_cap.release()

if cap is None or not cap.isOpened():
    print("Erro: Nenhuma câmera encontrada. Verifique a conexão.")
    exit(1)

def mouse_mover():
    """Thread daemon para mover o mouse continuamente"""
    while True:
        if mouse_control_enabled:
            with mouse_lock:
                x, y = mouse_target
            try:
                pyautogui.moveTo(x, y)
            except:
                pass
        time.sleep(0.01)

def detect_face_center(frame):
    """Detectar rosto e retornar centro (x, y, largura, altura)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        # Pega o maior rosto detectado
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        return (x + w//2, y + h//2, w, h)
    return None

# Inicia thread de movimento do mouse
threading.Thread(target=mouse_mover, daemon=True).start()

print("[INFO] Rastreamento com OpenCV Haar Cascade iniciado!")
print("[INFO] F7 = ligar/desligar mouse | C = calibrar | Q = sair")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Espelhar a imagem horizontalmente
    frame = cv2.flip(frame, 1)
    
    h, w, _ = frame.shape
    landmarks_frame = np.zeros_like(frame)
    
    # Detectar rosto
    face_info = detect_face_center(frame)
    
    if face_info is None:
        # Modo demonstração - sem rosto detectado
        cv2.putText(frame, "Procurando rosto... | F7: desabilitar mouse | Q: sair", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        with mouse_lock:
            mouse_target[0] = CENTER_X
            mouse_target[1] = CENTER_Y
    else:
        # Rosto detectado
        face_x, face_y, face_w, face_h = face_info
        
        # Desenhar retângulo ao redor do rosto
        cv2.rectangle(frame, (face_x - face_w//2, face_y - face_h//2), 
                     (face_x + face_w//2, face_y + face_h//2), (0, 255, 0), 2)
        cv2.circle(frame, (face_x, face_y), 5, (0, 0, 255), -1)
        
        # Calcular posição do rosto em relação ao centro
        # Quanto mais o rosto se move, mais o mouse segue
        yaw_norm = (face_x - w//2) / (w//2)  # -1 a 1 (esquerda a direita)
        pitch_norm = (face_y - h//2) / (h//2)  # -1 a 1 (cima a baixo)
        
        # Mapear para posição da tela
        screen_x = int(CENTER_X + yaw_norm * (CENTER_X * 0.8))
        screen_y = int(CENTER_Y + pitch_norm * (CENTER_Y * 0.8))
        
        # Clampar para limites da tela
        screen_x = max(10, min(MONITOR_WIDTH - 10, screen_x))
        screen_y = max(10, min(MONITOR_HEIGHT - 10, screen_y))
        
        # Atualizar posição alvo do mouse
        if mouse_control_enabled:
            with mouse_lock:
                mouse_target[0] = screen_x
                mouse_target[1] = screen_y
        
        # Mostrar informações
        status = "[Mouse ON]" if mouse_control_enabled else "[Mouse OFF]"
        cv2.putText(frame, f"{status} - F7: toggle | C: calibrar | Q: sair", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Pos Tela: ({screen_x}, {screen_y})", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Exibir frames
    cv2.imshow("Eye Tracker - OpenCV", frame)
    
    # Processar inputs do teclado
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):  # F7 não funciona bem com cv2.waitKey, usar 'F' como alternativa
        mouse_control_enabled = not mouse_control_enabled
        print(f"[Mouse Control] {'Enabled' if mouse_control_enabled else 'Disabled'}")
        time.sleep(0.3)
    elif key == ord('c'):
        if face_info is not None:
            calibration_offset_yaw = 180 - raw_yaw_deg
            calibration_offset_pitch = 180 - raw_pitch_deg
            print(f"[Calibrated] Offset Yaw: {calibration_offset_yaw}, Offset Pitch: {calibration_offset_pitch}")
        else:
            print("[Aviso] Calibração requer rosto detectado")
        time.sleep(0.3)
    
    # Detectar F7 com keyboard (funciona melhor)
    if keyboard.is_pressed('f7'):
        mouse_control_enabled = not mouse_control_enabled
        print(f"[Mouse Control] {'Enabled' if mouse_control_enabled else 'Disabled'}")
        time.sleep(0.3)

cap.release()
cv2.destroyAllWindows()
print("[INFO] Programa encerrado.")
