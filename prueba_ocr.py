import cv2
import easyocr
import numpy as np
from flask import Flask, Response, jsonify
from flask_cors import CORS
import threading
import time
import mysql.connector
from roboflow import Roboflow

# ====================================================================
# 1. CONFIGURACIONES PRINCIPALES
# ====================================================================

# 🔴 ¡IMPORTANTE! ACTUALIZADO PARA CONECTARSE A LA NUBE (AIVEN)
# Reemplaza 'TU_PASSWORD_DE_AIVEN' por la contraseña real que te dio Aiven
DB_CONFIG = {
    'host': 'central-esterilizacion-deineracosta2006.b.aivencloud.com',
    'user': 'avnadmin',       
    'password': 'AVNS_6M0I_jT6XTZh4Xn9Xn3',
    'database': 'defaultdb',
    'port': 16046
}

rf = Roboflow(api_key="RKVj1NLdzAL91w37CIsw")
workspace = rf.workspace("sistemaquirurgico")

modelo_forma = workspace.project("modelo-de-entrenamiento-z45wx").version(1).model
modelo_estado = workspace.project("sistema-quirurgico-djxkq").version(2).model
modelo_general = workspace.project("instrumental-oftalmologico-ztgvs").version(3).model

app = Flask(__name__)
CORS(app)

print("⏳ Cargando motor OCR...")
lector = easyocr.Reader(['en', 'es'])
print("✅ OCR Listo.")

camera = None
last_frame = None
lock = threading.Lock()
cajas_roboflow = [] 

def capture_camera():
    """Bucle Inmortal y Cazador de Puertos Automático"""
    global camera, last_frame
    
    while True:
        # Si no hay cámara conectada, activamos el radar para buscarla
        if camera is None or not camera.isOpened():
            print("🔍 Buscando microscopio en los puertos USB...")
            
            # Escaneamos los puertos del 0 al 4 automáticamente
            camara_encontrada = False
            for puerto_prueba in range(5):
                temp_cam = cv2.VideoCapture(puerto_prueba, cv2.CAP_DSHOW)
                if temp_cam.isOpened():
                    print(f"✅ ¡Microscopio detectado automáticamente en el puerto {puerto_prueba}!")
                    camera = temp_cam
                    camara_encontrada = True
                    break # Detenemos la búsqueda porque ya la encontramos
                else:
                    temp_cam.release()
            
            # Si revisó todos los puertos y no hay nada, espera y vuelve a buscar
            if not camara_encontrada:
                print("❌ Ningún microscopio conectado. Esperando cable USB...")
                time.sleep(2)
                continue

        # Una vez conectada, leemos el video
        success, frame = camera.read()
        
        if success and frame is not None:
            with lock:
                last_frame = frame.copy()
        else:
            # Si alguien desconecta el cable de golpe, se limpia y vuelve a buscar
            print("⚠️ Microscopio desconectado. Reiniciando radar de puertos...")
            if camera is not None:
                camera.release()
            camera = None
            time.sleep(1)
            continue

        time.sleep(0.03)

def roboflow_worker():
    """Consulta a Roboflow usando una copia exacta para NO congelar el video en vivo"""
    global last_frame, cajas_roboflow
    while True:
        if last_frame is not None:
            with lock:
                frame_para_ia = last_frame.copy()
                
            nuevas_cajas = []
            
            try:
                p_forma = modelo_forma.predict(frame_para_ia, confidence=40, overlap=30).json()
                if "predictions" in p_forma: nuevas_cajas.extend(p_forma["predictions"])
            except Exception as e: print(f"⚠️ Error IA Forma: {e}")
            
            try:
                p_estado = modelo_estado.predict(frame_para_ia, confidence=40, overlap=30).json()
                if "predictions" in p_estado: nuevas_cajas.extend(p_estado["predictions"])
            except Exception as e: print(f"⚠️ Error IA Estado: {e}")

            try:
                p_gen = modelo_general.predict(frame_para_ia, confidence=40, overlap=30).json()
                if "predictions" in p_gen: nuevas_cajas.extend(p_gen["predictions"])
            except Exception as e: print(f"⚠️ Error IA General: {e}")
            
            cajas_roboflow = nuevas_cajas
            
        time.sleep(1.0)

threading.Thread(target=capture_camera, daemon=True).start()
threading.Thread(target=roboflow_worker, daemon=True).start()

def obtener_codigos_db():
    try:
        conexion = mysql.connector.connect(**DB_CONFIG)
        cursor = conexion.cursor()
        cursor.execute("SELECT codigo FROM hojavidainstrumento") # Asegúrate de que coincida con tu tabla
        resultados = cursor.fetchall()
        conexion.close()
        return [fila[0] for fila in resultados]
    except Exception as e:
        print(f"❌ Error BD: {e}")
        return []

def buscar_match_inteligente(texto_ocr, codigos_reales):
    texto_limpio = texto_ocr.replace(" ", "").replace("_", "").upper()
    texto_sin_guion = texto_limpio.replace("-", "")
    
    for codigo_bd in codigos_reales:
        if codigo_bd.replace("-", "").upper() == texto_sin_guion: return codigo_bd

    texto_corregido = texto_sin_guion.replace("O", "0").replace("I", "1").replace("S", "5")
    for codigo_bd in codigos_reales:
        if codigo_bd.replace("-", "").upper().replace("O", "0") == texto_corregido:
            return codigo_bd

    if len(texto_corregido) >= 4:
        for codigo_bd in codigos_reales:
            if texto_corregido in codigo_bd.replace("-", "").upper().replace("O", "0"):
                return codigo_bd
    return None

def generate_video():
    global last_frame, cajas_roboflow
    while True:
        with lock:
            if last_frame is None:
                # Si no hay cámara aún, esperamos un poco
                time.sleep(0.1)
                continue
            frame_to_send = last_frame.copy()
        
        for caja in cajas_roboflow:
            x, y, w, h = caja['x'], caja['y'], caja['width'], caja['height']
            clase = caja['class']
            
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            
            color = (0, 255, 0) 
            if "oxido" in clase.lower() or "roto" in clase.lower() or "sangre" in clase.lower():
                color = (0, 0, 255)
                
            cv2.rectangle(frame_to_send, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_to_send, clase, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame_to_send)
        if ret:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/escanear', methods=['POST'])
def escanear():
    global last_frame, cajas_roboflow
    print("\n⏳ Vercel pidió escanear el metal...")
    
    with lock:
        if last_frame is None:
            return jsonify({"status": "error", "message": "Cámara no lista. Verifica que el cable USB esté conectado."})
        frame_actual = last_frame.copy()

    gris = cv2.cvtColor(frame_actual, cv2.COLOR_BGR2GRAY)
    contraste_alto = cv2.convertScaleAbs(gris, alpha=2.0, beta=0)
    imagen_limpia = cv2.bilateralFilter(contraste_alto, 11, 17, 17)

    resultados = lector.readtext(imagen_limpia)
    if len(resultados) == 0:
        print("❌ OCR no detectó ninguna letra.")
        return jsonify({"status": "error", "message": "Letras no legibles. Ajusta el enfoque."})

    texto_crudo = "".join([res[1] for res in resultados])
    print(f"✅ Lectura OCR: '{texto_crudo}'")

    codigos_reales = obtener_codigos_db()
    if not codigos_reales:
        return jsonify({"status": "error", "message": "Fallo en conexión con la base de datos."})

    codigo_perfecto = buscar_match_inteligente(texto_crudo, codigos_reales)

    if not codigo_perfecto:
        return jsonify({"status": "error", "message": f"Se leyó '{texto_crudo}' pero NO existe en BD."})

    estado_fisico = "Buen estado"
    forma_detectada = "No identificada"
    
    for caja in cajas_roboflow:
        clase = caja['class'].lower()
        if "oxido" in clase or "roto" in clase or "sangre" in clase:
            estado_fisico = "Mal estado"
        else:
            forma_detectada = caja['class']

    print(f"🎯 MATCH: {codigo_perfecto} | Visual: {forma_detectada} | Estado: {estado_fisico}")
    
    return jsonify({
        "status": "success",
        "codigo": codigo_perfecto,
        "forma_fisica": forma_detectada,
        "estado_fisico": estado_fisico
    })

if __name__ == '__main__':
    print("\n🚀 Motor Supremo (OCR + Roboflow + MySQL) Iniciado.")
    app.run(host='0.0.0.0', port=5000, debug=False)