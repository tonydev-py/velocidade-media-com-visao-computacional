import cv2
import torch
import gc
import time

modelo = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Parâmetros
distancia_real_metros = 6 #estimativa podendo ser maior ou menor
distancia_pixels = 101.92  
pixel_por_metro = distancia_pixels / distancia_real_metros

linha1_y, linha2_y = 91, 189
rastreamento, velocidades = {}, {}

def detectar_veiculos(video_path, output_path):
    global rastreamento, velocidades

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    largura, altura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (largura, altura))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = modelo(frame_rgb, size=640)

        
        cv2.line(frame, (0, linha2_y), (frame.shape[1], linha2_y), (0, 255, 0), 2)

        veiculos_detectados = []

        for *xyxy, conf, cls in resultados.xyxy[0]:
            if conf > 0.3 and modelo.names[int(cls)] == 'car':
                x1, y1, x2, y2 = map(int, xyxy)
                centro_x, centro_y = (x1 + x2) // 2, (y1 + y2) // 2

                veiculos_detectados.append(centro_x)

                if centro_x not in rastreamento and centro_y >= linha1_y:
                    rastreamento[centro_x] = time.time()

                elif centro_x in rastreamento and centro_x not in velocidades and centro_y >= linha2_y:
                    tempo_decorrido = time.time() - rastreamento[centro_x]
                    if tempo_decorrido > 0:
                        velocidades[centro_x] = round((distancia_real_metros / tempo_decorrido) * 3.6, 1)

                if centro_x in velocidades:
                    cv2.putText(frame, f"{velocidades[centro_x]:.1f} km/h", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        rastreamento = {k: v for k, v in rastreamento.items() if k in veiculos_detectados}
        velocidades = {k: v for k, v in velocidades.items() if k in veiculos_detectados}

        out.write(frame)
        cv2.imshow('Detecção de Veículos', frame)
        
        del frame, resultados
        gc.collect()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

video_path = "veiculos.mp4"
output_path = "resultado.mp4"
detectar_veiculos(video_path, output_path)
