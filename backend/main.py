import cv2
import asyncio
import json
import logging
import threading
import queue
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCIceCandidate
from aiortc.mediastreams import VideoFrame
from starlette.websockets import WebSocketDisconnect
import numpy as np
import time
from collections import deque, defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import onnxruntime as ort
import torch
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/prototype_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()
logger.info("FastAPI inicializado")

# Inicializar Firestore
try:
    cred = credentials.Certificate("proyecto-ia3-ff33c-firebase-adminsdk-fbsvc-cd48c40359.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    logger.info("Firestore inicializado correctamente")
except Exception as e:
    logger.error(f"Error al inicializar Firestore: {str(e)}")
    raise

# Cargar modelos
try:
    yolo_model = YOLO("models/best.pt")
    logger.info("Modelo YOLOv11 cargado correctamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo YOLOv11: {str(e)}")
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
CLIP_DURATION_SECONDS = 10
FPS = 15  # Ajustado para la Trust Taxon (puede ser 30 si usas 1080p)
CLIP_FRAMES = CLIP_DURATION_SECONDS * FPS
STRIDE_FRAMES = CLIP_FRAMES // 2
TIMESFORMER_FPS = 15
NUM_FRAMES_TIMESFORMER = 8
THRESHOLD_VIOLENCE = 0.7
YOLO_CONF_THRESHOLD = 0.6
YOLO_PROCESS_INTERVAL = 5

# Buffers y estado global
pcs = set()
frame_buffer = deque(maxlen=CLIP_FRAMES)
trajectories = defaultdict(list)
last_trajectories = []
violence_detected = False
last_prob_violence = 0.0
last_violence_ids = []
is_detecting = False
event_buffer = []
last_firestore_write = time.time()

# Cola para procesamiento en segundo plano
processing_queue = queue.Queue()

# Funciones de preprocesamiento y predicción
def preprocess_frame_for_yolo(frame, target_size=(416, 416)):
    h, w = frame.shape[:2]
    target_h, target_w = target_size
    ratio = min(target_w / w, target_h / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    padded_frame[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_frame
    return padded_frame

def preprocess_frames_for_timesformer(frames):
    total_frames = len(frames)
    frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES_TIMESFORMER, dtype=int)
    final_frames = [frames[i] for i in frame_indices]
    processed_frames = np.zeros((NUM_FRAMES_TIMESFORMER, 224, 224, 3), dtype=np.float32)
    for i, frame in enumerate(final_frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        ratio = min(224 / w, 224 / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_top = (224 - new_h) // 2
        pad_left = (224 - new_w) // 2
        processed_frames[i, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_frame
    processed_frames = processed_frames.transpose(0, 3, 1, 2)
    processed_frames = (processed_frames / 255.0).astype(np.float32)
    processed_frames = np.expand_dims(processed_frames, axis=0)
    return processed_frames

def predict_violence(frames):
    try:
        pixel_values = preprocess_frames_for_timesformer(frames)
        outputs = timesformer_session.run(None, {"pixel_values": pixel_values})[0]
        logits = torch.tensor(outputs)
        probs = torch.softmax(logits, dim=1)
        prob_violence = probs[0, 1].item()
        pred = 1 if prob_violence > THRESHOLD_VIOLENCE else 0
        if 0.45 < prob_violence < 0.55:
            pred = 0
        return pred, prob_violence
    except Exception as e:
        logger.error(f"Error al predecir violencia: {str(e)}")
        return 0, 0.0

def get_ids_in_interval(start_frame, end_frame):
    ids_in_interval = set()
    for frame_num in range(start_frame, end_frame + 1):
        if frame_num in trajectories:
            for track_id, _ in trajectories[frame_num]:
                ids_in_interval.add(track_id)
    return sorted(list(ids_in_interval))

# Procesamiento en segundo plano
def process_frames_worker():
    global violence_detected, last_prob_violence, last_violence_ids, last_trajectories, event_buffer, last_firestore_write
    while True:
        try:
            frame_data = processing_queue.get()
            if frame_data is None:
                break

            frame, frame_count = frame_data
            yolo_frame = preprocess_frame_for_yolo(frame, target_size=(416, 416))
            results = yolo_model(yolo_frame, conf=YOLO_CONF_THRESHOLD, classes=0, iou=0.5)
            detections = []
            scale_x = frame.shape[1] / 416
            scale_y = frame.shape[0] / 416
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = box
                    x1, x2 = x1 * scale_x, x2 * scale_x
                    y1, y2 = y1 * scale_y, y2 * scale_y
                    detections.append(([x1, y1, x2 - x1, y2 - y1], score, 0))

            tracks = deepsort.update_tracks(detections, frame=frame)
            current_trajectories = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                current_trajectories.append((track_id, ltrb))
            trajectories[frame_count] = current_trajectories
            last_trajectories = current_trajectories

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
                    event_buffer.append(event)
                    logger.info(f"Violencia detectada: {event}")

                    if time.time() - last_firestore_write >= 10:
                        for event in event_buffer:
                            db.collection("violence_events").add(event)
                        event_buffer.clear()
                        last_firestore_write = time.time()

            processing_queue.task_done()
        except Exception as e:
            logger.error(f"Error en procesamiento en segundo plano: {str(e)}")
            processing_queue.task_done()

# Iniciar el hilo de procesamiento
processing_thread = threading.Thread(target=process_frames_worker, daemon=True)
processing_thread.start()

# Clase para procesar y transmitir video
class VideoTransformTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.frame_count = 0
        self.connect_attempts = 0
        self.max_attempts = 5
        self.connect()

    def connect(self):
        global is_detecting
        while self.connect_attempts < self.max_attempts:
            # Usar cámara USB (índice 0, ajusta si es necesario)
            self.cap = cv2.VideoCapture(1)
            if self.cap.isOpened():
                # Configurar resolución y FPS
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 1080p para mejor rendimiento
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 15)
                logger.info("Cámara USB (Trust Taxon 2K QHD) abierta correctamente")
                is_detecting = False
                return
            logger.warning(f"Intento {self.connect_attempts + 1} fallido para abrir la cámara USB")
            self.connect_attempts += 1
            time.sleep(2)
        logger.error("No se pudo abrir la cámara USB después de varios intentos")
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(self.frame, "Cámara no disponible", (150, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    async def recv(self):
        global violence_detected, last_prob_violence, last_violence_ids, last_trajectories, is_detecting, frame_buffer
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Error al leer frame de la cámara USB")
                self.cap.release()
                self.connect()
                return await self.recv()

            self.frame_count += 1
            frame_buffer.append(frame.copy())

            if is_detecting and self.frame_count % YOLO_PROCESS_INTERVAL == 0:
                processing_queue.put((frame.copy(), self.frame_count))

            for track_id, ltrb in last_trajectories:
                x1, y1, x2, y2 = map(int, ltrb)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            status_text = f"Violencia: {'Sí' if violence_detected else 'No'} (Prob: {last_prob_violence:.4f})"
            status_color = (0, 0, 255) if violence_detected else (0, 255, 0)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            if violence_detected and last_violence_ids:
                ids_text = f"IDs Involucrados: {last_violence_ids}"
                cv2.putText(frame, ids_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            frame = self.frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(frame, dtype=np.uint8)
        return VideoFrame.from_ndarray(frame, format="rgb24")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("icecandidate")
    async def on_icecandidate(candidate):
        if candidate:
            logger.info(f"Enviando ICE candidate al frontend: {candidate}")
            await websocket.send_json({"candidate": candidate.to_json()})

    @pc.on("track")
    def on_track(track):
        logger.info(f"Track recibido: {track.kind}")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Estado de conexión WebRTC: {pc.connectionState}")

    video_track = VideoTransformTrack()
    pc.addTrack(video_track)

    try:
        while True:
            data = await websocket.receive_json()
            logger.info(f"Datos recibidos del frontend: {data}")
            if "offer" in data:
                offer = RTCSessionDescription(sdp=data["offer"]["sdp"], type=data["offer"]["type"])
                await pc.setRemoteDescription(offer)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await websocket.send_json({
                    "answer": {
                        "sdp": pc.localDescription.sdp,
                        "type": pc.localDescription.type
                    }
                })
            elif "candidate" in data and data["candidate"]:
                try:
                    candidate_sdp = data["candidate"]["candidate"]
                    candidate = RTCIceCandidate(
                        sdp=candidate_sdp,
                        sdpMid=data["candidate"]["sdpMid"],
                        sdpMLineIndex=data["candidate"]["sdpMLineIndex"]
                    )
                    await pc.addIceCandidate(candidate)
                    logger.info(f"Candidato ICE añadido: {candidate_sdp}")
                except Exception as e:
                    logger.error(f"Error al procesar candidato ICE: {str(e)}")
            elif "start_detection" in data:
                global is_detecting
                is_detecting = True
                logger.info("Detección iniciada")
            elif "stop_detection" in data:
                is_detecting = False
                logger.info("Detección detenida")
    except WebSocketDisconnect:
        logger.info("Cliente desconectado")
        pcs.discard(pc)
        await pc.close()
    except Exception as e:
        logger.error(f"Error en WebSocket: {str(e)}")
        pcs.discard(pc)
        await pc.close()

@app.get("/")
async def get():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
        <head>
            <title>WebRTC Video Stream</title>
        </head>
        <body>
            <h1>Prototipo de Streaming WebRTC</h1>
            <video id="video" autoplay playsinline></video>
            <script>
                async function start() {
                    const ws = new WebSocket("ws://192.168.1.6:8000/ws");
                    const pc = new RTCPeerConnection({
                        iceServers: [
                            { urls: "stun:stun.l.google.com:19302" },
                        ]
                    });
                    const video = document.getElementById("video");

                    pc.addTransceiver('video', { direction: 'recvonly' });

                    pc.ontrack = (event) => {
                        if (event.streams[0]) {
                            video.srcObject = event.streams[0];
                        }
                    };

                    pc.onicecandidate = (event) => {
                        if (event.candidate) {
                            ws.send(JSON.stringify({ candidate: event.candidate }));
                        }
                    };

                    pc.onconnectionstatechange = () => {
                        console.log("Estado de conexión WebRTC:", pc.connectionState);
                    };

                    ws.onmessage = async (event) => {
                        const data = JSON.parse(event.data);
                        if (data.answer) {
                            await pc.setRemoteDescription(new RTCSessionDescription(data.answer));
                        } else if (data.candidate) {
                            await pc.addIceCandidate(new RTCIceCandidate(data.candidate));
                        }
                    };

                    ws.onopen = async () => {
                        const offer = await pc.createOffer();
                        await pc.setLocalDescription(offer);
                        ws.send(JSON.stringify({ offer }));
                    };

                    ws.onerror = (error) => {
                        console.error("WebSocket error:", error);
                    };
                }

                window.onload = start;
            </script>
        </body>
    </html>
    """)