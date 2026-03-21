import cv2
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import time
import threading
import numpy as np
import os
from ultralytics import YOLO
import easyocr # <-- NUEVA IMPORTACIÓN PARA LEER METAL

# --- CONFIGURACIÓN GENERAL ---
VIDEO_SOURCE = 0 # Usa 1, 0, o 2 (el número que te haya funcionado en la prueba anterior)
MODEL_WEIGHTS = 'yolov8n.pt' 

# --- INICIALIZAR LECTOR OCR (Para leer códigos en el metal) ---
print("⏳ Cargando modelo OCR (EasyOCR)... Puede tardar unos segundos.")
try:
    lector_ocr = easyocr.Reader(['en', 'es'])
    print("✅ Modelo OCR cargado correctamente.")
except Exception as e:
    print(f"❌ Error al cargar OCR: {e}")

# --- DICCIONARIO DE TRADUCCIÓN ---
# Esto traduce las clases comunes del dataset COCO al español.
CLASS_TRANSLATIONS = {
    'person': 'persona', 'bicycle': 'bicicleta', 'car': 'coche', 'motorcycle': 'motocicleta',
    'airplane': 'avión', 'bus': 'autobús', 'train': 'tren', 'truck': 'camión', 'boat': 'barco',
    'traffic light': 'semáforo', 'fire hydrant': 'boca de incendios', 'stop sign': 'señal de stop',
    'parking meter': 'parquímetro', 'bench': 'banco', 'bird': 'pájaro', 'cat': 'gato', 'dog': 'perro',
    'horse': 'caballo', 'sheep': 'oveja', 'cow': 'vaca', 'elephant': 'elefante', 'bear': 'oso',
    'zebra': 'cebra', 'giraffe': 'jirafa', 'backpack': 'mochila', 'umbrella': 'paraguas',
    'handbag': 'bolso', 'tie': 'corbata', 'suitcase': 'maleta', 'frisbee': 'frisbi', 'skis': 'esquís',
    'snowboard': 'tabla de nieve', 'sports ball': 'balón deportivo', 'kite': 'cometa',
    'baseball bat': 'bate de béisbol', 'baseball glove': 'guante de béisbol', 'skateboard': 'patineta',
    'surfboard': 'tabla de surf', 'tennis racket': 'raqueta de tenis', 'bottle': 'botella',
    'wine glass': 'copa de vino', 'cup': 'taza', 'fork': 'tenedor', 'knife': 'cuchillo',
    'spoon': 'cuchara', 'bowl': 'cuenco', 'banana': 'plátano', 'apple': 'manzana', 'sandwich': 'sándwich',
    'orange': 'naranja', 'broccoli': 'brócoli', 'carrot': 'zanahoria', 'hot dog': 'perrito caliente',
    'pizza': 'pizza', 'donut': 'dónut', 'cake': 'tarta', 'chair': 'silla', 'couch': 'sofá',
    'potted plant': 'planta en maceta', 'bed': 'cama', 'dining table': 'mesa de comedor',
    'toilet': 'inodoro', 'tv': 'televisión', 'laptop': 'portátil', 'mouse': 'ratón',
    'remote': 'mando a distancia', 'keyboard': 'teclado', 'cell phone': 'teléfono móvil',
    'microwave': 'microondas', 'oven': 'horno', 'toaster': 'tostadora', 'sink': 'fregadero',
    'refrigerator': 'nevera', 'book': 'libro', 'clock': 'reloj', 'vase': 'florero',
    'scissors': 'tijeras', 'teddy bear': 'oso de peluche', 'hair drier': 'secador de pelo',
    'toothbrush': 'cepillo de dientes', 'cacao': 'cacao', 'frasco': 'frasco'
}

# --- Inicializar YOLO ---
model_yolo = None
try:
    model_yolo = YOLO(MODEL_WEIGHTS)
    print(f"✅ Modelo YOLOv8 '{MODEL_WEIGHTS}' cargado.")
except Exception as e:
    print(f"❌ Error CRÍTICO al cargar el modelo YOLO: {e}")

# --- ESTADO GLOBAL ---
global_system_state = 0
latest_analysis_results = {"estado_sistema": 0, "detecciones": [], "camera_status": "connected"}
last_processed_frame_for_display = None
analysis_lock = threading.Lock()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secreto'
socketio = SocketIO(app, async_mode='threading')
camera = None
camera_thread = None
stop_camera_thread = False

# --- LÓGICA PRINCIPAL DEL VIDEO ---
def process_camera_feed():
    """Hilo principal para capturar frames y realizar la detección YOLO."""
    global latest_analysis_results, last_processed_frame_for_display, camera
    
    print("Iniciando hilo de cámara para Detección General con traducción...")
    frame_count = 0
    
    while not stop_camera_thread:
        if camera is None or not camera.isOpened():
            time.sleep(2)
            try:
                if camera: camera.release()
                camera = cv2.VideoCapture(VIDEO_SOURCE)
                time.sleep(1)
                if not camera.isOpened():
                    with analysis_lock: latest_analysis_results["camera_status"] = "disconnected"
                    socketio.emit('status_update', latest_analysis_results)
                    continue
                with analysis_lock: latest_analysis_results["camera_status"] = "connected"
            except: continue

        success, frame = camera.read()
        if not success:
            print(f"⚠️ ADVERTENCIA: No se recibe imagen en el índice {VIDEO_SOURCE}. ¿El TOMLOV está en modo 'USB UVC' o el índice es incorrecto?")
            time.sleep(2)
            continue

        frame_count += 1
        
        if frame_count % 5 == 0 and model_yolo:
            frame_to_process = frame.copy()
            
            try:
                results = model_yolo(frame_to_process, verbose=False, conf=0.4, iou=0.3)
                annotated_frame = frame_to_process.copy()
                current_detections_es = set()
                
                for r in results:
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        class_name_en = model_yolo.names[class_id]
                        class_name_es = CLASS_TRANSLATIONS.get(class_name_en, class_name_en) 
                        
                        current_detections_es.add(class_name_es)
                        
                        x, y, w, h = int(box.xywh[0][0]), int(box.xywh[0][1]), int(box.xywh[0][2]), int(box.xywh[0][3])
                        conf = float(box.conf[0])
                        
                        x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
                        color = (0, 255, 0) 
                        
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{class_name_es} {conf:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                with analysis_lock:
                    latest_analysis_results = {
                        "estado_sistema": 0, 
                        "detecciones": sorted(list(current_detections_es)), 
                        "camera_status": "connected"
                    }
                    last_processed_frame_for_display = annotated_frame.copy()
                    
                socketio.emit('status_update', latest_analysis_results)

            except Exception as e: 
                print(f"Error en el procesamiento YOLO: {e}")
                annotated_frame = frame_to_process.copy()
                cv2.putText(annotated_frame, "Error YOLO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                with analysis_lock:
                     last_processed_frame_for_display = annotated_frame.copy()
        
        else:
             with analysis_lock:
                 if last_processed_frame_for_display is None:
                     last_processed_frame_for_display = frame.copy()

        time.sleep(0.01)

def generate_frames_display():
    while not stop_camera_thread:
        with analysis_lock: frame = last_processed_frame_for_display
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ret: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)


# ======================================================================
# --- NUEVA RUTA PARA LEER EL METAL AL PRESIONAR EL BOTÓN (OCR) ---
# ======================================================================
@app.route('/detectar_codigo', methods=['POST'])
def detectar_codigo_route():
    global last_processed_frame_for_display
    
    # 1. Tomar la foto actual en alta calidad del flujo de video
    with analysis_lock:
        if last_processed_frame_for_display is None:
            return jsonify({"status": "error", "message": "La cámara aún no está lista."})
        frame_actual = last_processed_frame_for_display.copy()

    # 2. Leer el texto con EasyOCR
    print("\n⏳ Escaneando el instrumento...")
    resultados = lector_ocr.readtext(frame_actual)
    
    # 3. Limpiar la "basura": Unimos todo, en mayúsculas y sin espacios
    texto_unido = "".join([res[1] for res in resultados]).upper().replace(" ", "").replace("_", "")
    print(f"Texto detectado y limpiado por OCR: '{texto_unido}'")

    # 4. BASE DE DATOS SIMULADA (Filtros inteligentes a prueba de errores)
    instrumento_detectado = "Desconocido (No registrado)"
    
    # Verificamos si fragmentos del código están en lo que leyó la IA
    if "0829" in texto_unido or "OFCBO" in texto_unido:
        instrumento_detectado = "Pinza de Disección (0829-OFCBO)"
    elif "GLP" in texto_unido or "POO" in texto_unido or "P00" in texto_unido:
        instrumento_detectado = "Pinza Fina (OF GL PO004)"
    elif "1508" in texto_unido or "RUMEX" in texto_unido or "R0" in texto_unido:
        instrumento_detectado = "Mango de Bisturí Rumex"

    print(f"Resultado final: {instrumento_detectado}\n")

    # 5. Enviamos el resultado al Frontend (Dashboard)
    return jsonify({
        "status": "success", 
        "instrumento": instrumento_detectado,
        "codigo_crudo": texto_unido
    })
# ======================================================================

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames_display(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status_data')
def status_data():
    with analysis_lock: return jsonify(latest_analysis_results)

@app.route('/reset', methods=['POST'])
def reset_system_route():
    global global_system_state
    with analysis_lock:
        global_system_state = 0
    socketio.emit('status_update', {"estado_sistema": 0, "detecciones": []})
    return jsonify({"status": "reset"})

@socketio.on('connect')
def handle_connect():
    with analysis_lock: emit('status_update', latest_analysis_results)

if __name__ == '__main__':
    print("--------------------------------------------------------------------")
    print("Iniciando Sistema de Detección General y Lector OCR...")
    print("--------------------------------------------------------------------")
    
    stop_camera_thread = False
    camera_thread = threading.Thread(target=process_camera_feed)
    camera_thread.daemon = True
    camera_thread.start()

    socketio.run(app, debug=True, host='0.0.0.0', port=5000, use_reloader=False, allow_unsafe_werkzeug=True)

    stop_camera_thread = True
    if camera_thread: camera_thread.join()