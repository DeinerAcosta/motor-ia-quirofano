from ultralytics import YOLO

def train_model():

    model = YOLO('yolov8n.pt')

    print("Iniciando el entrenamiento del modelo...")
    results = model.train(data='dataset/data.yaml', epochs=50, imgsz=640, project='runs', name='instrument_detector')
    
    print("¡Entrenamiento completado exitosamente!")
    print(f"El modelo entrenado se encuentra en la carpeta: runs/instrument_detector/weights/best.pt")

if __name__ == '__main__':
    train_model()



