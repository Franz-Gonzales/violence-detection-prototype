import cv2
import numpy as np
from collections import deque, defaultdict
from flask import Flask, Response, send_file
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import onnxruntime as ort
import torch
import os
from datetime import datetime
import logging
import firebase_admin
from firebase_admin import credentials, firestore

# Ensure the logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging with detailed format and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/prototype_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Inicializar Flask y SocketIO
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")
logger.info("Flask y SocketIO inicializados")

# Inicializar Firestore
try:
    cred = credentials.Certificate("proyecto-ia3-ff33c-firebase-adminsdk-fbsvc-cd48c40359.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    logger.info("Firestore inicializado correctamente")
except Exception as e:
    logger.error(f"Error al inicializar Firestore: {str(e)}")
    raise

# Cargar modelos con manejo de errores
try:
    yolo_model = YOLO("models/best.pt")
    logger.info("Modelo YOLOv8 cargado correctamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo YOLOv8: {str(e)}")
    raise

try:
    deepsort = DeepSort(max_age=30, n_init=3)
    logger.info("DeepSort inicializado correctamente")
except Exception as e:
    logger.error(f"Error al inicializar DeepSort: {str(e)}")
    raise

try:
    timesformer_session = ort.InferenceSession("models/timesformer_finetuned.onnx")
    logger.info("Modelo TimeSformer cargado correctamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo TimeSformer: {str(e)}")
    raise

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Usando dispositivo: {device}")

# Parámetros ajustados
CLIP_DURATION_SECONDS = 10  # Duración del clip para TimeSformer (segundos)
FPS = 30  # FPS objetivo inicial (se ajustará según la cámara)
CLIP_FRAMES = CLIP_DURATION_SECONDS * FPS  # Se ajustará después
STRIDE_FRAMES = CLIP_FRAMES // 5  # Frecuencia de predicción de TimeSformer (cada 2 segundos)
TIMESFORMER_FPS = 15  # FPS para muestrear frames para TimeSformer
NUM_FRAMES_TIMESFORMER = 8  # Número de frames por clip para TimeSformer
THRESHOLD_VIOLENCE = 0.7  # Umbral para clasificar violencia
YOLO_CONF_THRESHOLD = 0.6  # Umbral de confianza para YOLOv8
YOLO_PROCESS_INTERVAL = 3  # Procesar YOLO y DeepSort cada 3 frames

# Buffers y estado global
frame_buffer = deque(maxlen=CLIP_FRAMES)  # Buffer para almacenar frames
trajectories = defaultdict(list)  # Almacenar trayectorias de personas
last_trajectories = []  # Almacenar las últimas trayectorias para dibujar en cada frame
violence_detected = False  # Estado de detección de violencia
last_prob_violence = 0.0  # Última probabilidad de violencia
last_violence_ids = []  # IDs involucrados en el último evento de violencia
is_detecting = False  # Estado para controlar la detección

# Función para preprocesar frames para YOLOv11 (manteniendo relación de aspecto)
def preprocess_frame_for_yolo(frame, target_size=(640, 640)):
    """
    Preprocesa un frame para YOLOv11, redimensionando con padding para preservar la relación de aspecto.
    
    Args:
        frame (np.ndarray): Frame de entrada (H, W, 3).
        target_size (tuple): Tamaño objetivo (alto, ancho), por defecto (640, 640).
    
    Returns:
        np.ndarray: Frame preprocesado de tamaño target_size.
    """
    h, w = frame.shape[:2]
    target_h, target_w = target_size
    ratio = min(target_w / w, target_h / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    padded_frame[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_frame
    return padded_frame

# Función para preprocesar frames para TimeSformer
def preprocess_frames_for_timesformer(frames):
    """
    Preprocesa una secuencia de frames para TimeSformer, redimensionando con padding y normalizando.
    
    Args:
        frames (list): Lista de frames (H, W, 3).
    
    Returns:
        np.ndarray: Tensor preprocesado de tamaño (1, T, C, H, W).
    """
    total_frames = len(frames)
    frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES_TIMESFORMER, dtype=int)
    final_frames = [frames[i] for i in frame_indices]
    processed_frames = np.zeros((NUM_FRAMES_TIMESFORMER, 224, 224, 3), dtype=np.float32)
    for i, frame in enumerate(final_frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        ratio = min(224 / w, 224 / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pad_top = (224 - new_h) // 2
        pad_left = (224 - new_w) // 2
        processed_frames[i, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_frame
    processed_frames = processed_frames.transpose(0, 3, 1, 2)
    processed_frames = (processed_frames / 255.0).astype(np.float32)
    processed_frames = np.expand_dims(processed_frames, axis=0)
    return processed_frames

# Función para predecir violencia con TimeSformer (ONNX)
def predict_violence(frames):
    """
    Predice si un clip contiene violencia usando TimeSformer.
    
    Args:
        frames (list): Lista de frames del clip.
    
    Returns:
        tuple: (predicción binaria, probabilidad de violencia).
    """
    try:
        pixel_values = preprocess_frames_for_timesformer(frames)
        outputs = timesformer_session.run(None, {"pixel_values": pixel_values})[0]
        logits = torch.tensor(outputs)
        probs = torch.softmax(logits, dim=1)
        prob_violence = probs[0, 1].item()
        pred = 1 if prob_violence > THRESHOLD_VIOLENCE else 0
        if 0.45 < prob_violence < 0.55:
            pred = 0  # Zona de incertidumbre
        return pred, prob_violence
    except Exception as e:
        logger.error(f"Error al predecir violencia: {str(e)}")
        return 0, 0.0

# Función para obtener IDs en un intervalo de frames
def get_ids_in_interval(start_frame, end_frame):
    """
    Obtiene los IDs de personas presentes en un intervalo de frames.
    
    Args:
        start_frame (int): Frame inicial.
        end_frame (int): Frame final.
    
    Returns:
        list: Lista ordenada de IDs.
    """
    ids_in_interval = set()
    for frame_num in range(start_frame, end_frame + 1):
        if frame_num in trajectories:
            for track_id, _ in trajectories[frame_num]:
                ids_in_interval.add(track_id)
    return sorted(list(ids_in_interval))

# Procesamiento de video en tiempo real
def process_video():
    """
    Procesa el video en tiempo real, detectando personas con YOLOv11, siguiendo con DeepSort,
    y detectando violencia con TimeSformer.
    """
    logger.info("Iniciando process_video")
    cap = cv2.VideoCapture("http://7156-2800-cd0-c332-9400-34f8-c9f8-bce9-e00e.ngrok-free.app/video")
    # cap = cv2.VideoCapture("/root/violence-detection-prototype/backend/videos/fight_0620_004.mp4")
    # cap = cv2.VideoCapture("http://192.168.1.4:8080/video")
    if not cap.isOpened():
        logger.error("No se pudo abrir la cámara. Estado de apertura: %s", str(cap.isOpened()))
        return
    logger.info("Cámara abierta exitosamente")

    # Obtener FPS real de la cámara
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # Fallback a 25 si no se puede obtener
    logger.info(f"FPS real de la cámara: {actual_fps}")
    global FPS, CLIP_FRAMES, STRIDE_FRAMES
    FPS = actual_fps
    CLIP_FRAMES = int(CLIP_DURATION_SECONDS * FPS)
    STRIDE_FRAMES = CLIP_FRAMES // 5
    logger.info(f"Parámetros ajustados - FPS: {FPS}, CLIP_FRAMES: {CLIP_FRAMES}, STRIDE_FRAMES: {STRIDE_FRAMES}")

    frame_count = 0
    global violence_detected, last_prob_violence, last_violence_ids, is_detecting, last_trajectories

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("No se pudo capturar el frame")
                break

            frame_count += 1
            frame_buffer.append(frame.copy())

            if is_detecting:
                if frame_count % YOLO_PROCESS_INTERVAL == 0:
                    yolo_frame = preprocess_frame_for_yolo(frame, target_size=(640, 640))

                    # Detección con YOLOv8
                    results = yolo_model(yolo_frame, conf=YOLO_CONF_THRESHOLD, classes=0, iou=0.5)
                    detections = []
                    scale_x = frame.shape[1] / 640
                    scale_y = frame.shape[0] / 640
                    for result in results:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        scores = result.boxes.conf.cpu().numpy()
                        for box, score in zip(boxes, scores):
                            x1, y1, x2, y2 = box
                            x1, x2 = x1 * scale_x, x2 * scale_x
                            y1, y2 = y1 * scale_y, y2 * scale_y
                            detections.append(([x1, y1, x2 - x1, y2 - y1], score, 0))

                    # Seguimiento con DeepSort
                    tracks = deepsort.update_tracks(detections, frame=frame)
                    current_trajectories = []
                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                        track_id = track.track_id
                        ltrb = track.to_ltrb()
                        current_trajectories.append((track_id, ltrb))
                    trajectories[frame_count] = current_trajectories
                    last_trajectories = current_trajectories  # Actualizar las últimas trayectorias

                    # Predecir violencia con TimeSformer
                    if frame_count % STRIDE_FRAMES == 0 and len(frame_buffer) == CLIP_FRAMES:
                        start_frame = max(1, frame_count - CLIP_FRAMES + 1)
                        end_frame = frame_count
                        pred, prob_violence = predict_violence(list(frame_buffer))
                        violence_detected = (pred == 1)
                        last_prob_violence = prob_violence
                        ids_in_interval = get_ids_in_interval(start_frame, end_frame)
                        last_violence_ids = ids_in_interval if violence_detected else []

                        if violence_detected:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            event = {
                                "timestamp": timestamp,
                                "start_frame": start_frame,
                                "end_frame": end_frame,
                                "probability": prob_violence,
                                "ids_involved": ids_in_interval
                            }
                            db.collection("violence_events").add(event)
                            socketio.emit("violence_detected", event)
                            logger.info(f"Violencia detectada: {event}")

                # Dibujar bounding boxes usando las últimas trayectorias
                for track_id, ltrb in last_trajectories:
                    x1, y1, x2, y2 = map(int, ltrb)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Siempre dibujar el texto de estado y los IDs involucrados
            status_text = f"Violencia: {'Sí' if violence_detected else 'No'} (Prob: {last_prob_violence:.4f})"
            status_color = (0, 0, 255) if violence_detected else (0, 255, 0)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            if violence_detected and last_violence_ids:
                ids_text = f"IDs Involucrados: {last_violence_ids}"
                cv2.putText(frame, ids_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Enviar frame al frontend en resolución original con calidad ajustada
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                frame_bytes = buffer.tobytes()
                socketio.emit("frame", frame_bytes)

    except Exception as e:
        logger.error(f"Error en el procesamiento de video: {str(e)}")
    finally:
        cap.release()
        logger.info("Cámara liberada")

# Iniciar procesamiento en un hilo separado al conectar
@socketio.on("connect")
def handle_connect():
    logger.info("Cliente conectado")
    socketio.start_background_task(process_video)

# Manejar el inicio de la detección
@socketio.on("start_detection")
def handle_start_detection():
    global is_detecting
    is_detecting = True
    logger.info("Detección iniciada")

# Manejar la parada de la detección
@socketio.on("stop_detection")
def handle_stop_detection():
    global is_detecting
    is_detecting = False
    logger.info("Detección detenida")

# Ruta para servir la página web
@app.route("/")
def index():
    return app.send_static_file("index.html")

if __name__ == "__main__":
    logger.info("Iniciando servidor Flask en puerto 5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)