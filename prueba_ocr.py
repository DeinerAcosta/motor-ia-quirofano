import cv2
import easyocr

print("Cargando inteligencia artificial de lectura (puede tardar unos segundos)...")
# Inicializamos el lector en inglés (en) y español (es)
lector = easyocr.Reader(['en', 'es'])

# Inicia la cámara de tu microscopio TOMLOV
cap = cv2.VideoCapture(0) # <-- CAMBIA ESTE NÚMERO POR EL CORRECTO (0, 1 o 2)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print("¡Listo! Enfoca el texto grabado en el bisturí.")
print("Presiona la tecla ESPACIO para escanear, o 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mostramos la imagen en vivo
    cv2.imshow("Escaner de Instrumentos", frame)
    tecla = cv2.waitKey(1) & 0xFF

    # Si presionas la barra espaciadora, toma la foto y lee el texto
    if tecla == ord(' '):
        print("\n⏳ Escaneando el metal, por favor espera...")
        
        # Aquí ocurre la magia de EasyOCR
        resultados = lector.readtext(frame)
        
        print("--- RESULTADOS ---")
        if len(resultados) == 0:
            print("No se detectó ningún texto. Intenta mejorar la luz o el enfoque.")
        else:
            for (caja, texto, confianza) in resultados:
                print(f"Texto detectado: '{texto}' (Precisión: {confianza:.2f})")
        print("------------------\n")

    # Si presionas la 'q', se cierra
    elif tecla == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()