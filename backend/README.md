# Backend para Detección de Violencia Física Escolar

## Instalación

1. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configurar Firestore**:
   - Descarga las credenciales de Google Cloud Firestore desde tu proyecto en Google Cloud.
   - Coloca el archivo JSON de credenciales en `backend/` y actualiza la ruta en `app.py` (línea `cred = credentials.Certificate("path/to/firestore-credentials.json")`).

3. **Configurar la Cámara**:
   - Descarga la app **IP Webcam** en tu celular Android desde Google Play Store.
   - Inicia el servidor de streaming y anota la URL (por ejemplo, `http://192.168.1.X:8080/video`).
   - Actualiza la URL en `app.py` (línea `cap = cv2.VideoCapture("http://192.168.1.X:8080/video")`).

4. **Colocar los Modelos**:
   - Asegúrate de que `best.pt` y `timesformer_finetuned.onnx` estén en `backend/models/`.

## Ejecución

1. **Iniciar el servidor**:
   ```bash
   python app.py
   ```

2. **Acceder a la aplicación**:
   - Abre tu navegador y ve a `http://localhost:5000`.

## Notas
- Asegúrate de que el frontend esté compilado y los archivos estén en `backend/static/` (ver instrucciones en el README del frontend).
- Si TimeSformer es lento, considera reducir la resolución del video o el número de frames por clip en `app.py`.







## Here are the commands to manually install each library used in the provided code:

- Install OpenCV (cv2): pip install opencv-python
- Install NumPy (np): pip install numpy
- Install Flask: pip install flask
- Install Flask-SocketIO: pip install flask-socketio
- Install Ultralytics YOLO: pip install ultralytics
- Install DeepSORT Realtime: pip install deep-sort-realtime
- Install ONNX Runtime: pip install onnxruntime-gpu

- Install PyTorch: Follow the instructions at https://pytorch.org/get-started/locally/ to install the correct version for your system.
- Install Firebase Admin SDK: pip install firebase-admin
- pip install logging



# PONER EN ENTORNO VIRTUAL
- python3 -m venv venv
- source venv/bin/activate 
- venv\Scripts\activate -> para windows


# PARA EJECUTAR CON VENV: EN RUN-POD
- scp -P 40184 app.py root@213.192.2.78:/root/
- nohup python3 /root/app.py &  -> ejecutar en segundo plano
- cat nohup.out  -> verificar los logs