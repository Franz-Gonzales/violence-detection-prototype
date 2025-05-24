import cv2

# Abrir la cámara (índice 0 para la primera cámara USB, puede ser 1 o 2 si tienes otras cámaras)
cap = cv2.VideoCapture(2)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

# Configurar resolución (opcional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame")
        break

    # Mostrar el frame
    cv2.imshow("Trust Taxon 2K QHD Webcam Test", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()