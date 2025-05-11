# Crear el directorio de front
npx create-react-app violence-detection-web
cd violence-detection-web
npm install socket.io-client @mui/material @emotion/react @emotion/styled firebase

# Instalar las librerias
- pip install opencv-python ultralytics deepsort-realtime transformers torch flask flask-socketio firebase-admin numpy


## Configuración

1. Copia `backend/proyecto-ia3-ff33c-firebase-adminsdk-example.json` a `backend/proyecto-ia3-ff33c-firebase-adminsdk-fbsvc-cd48c40359.json`
2. Actualiza el archivo con tus credenciales de Firebase

# Para la cámara 
ngrok config add-authtoken <TU_AUTHTOKEN>
# EN la pagina obten tu token
ngrok config add-authtoken 2aBcDeFgHiJkLmNoPqRsTuVwXyZ_1234567890abcdefg`

# Configurar el ngok para el servidor de la cámara del celular
- ngrok http http://192.168.1.4:8080
- Forwarding    http://xyz9876.ngrok.io -> http://192.168.1.6:8080

- cap = cv2.VideoCapture("http://ecdc-2800-cd0-c332-9400-34f8-c9f8-bce9-e00e.ngrok-free.app/video")