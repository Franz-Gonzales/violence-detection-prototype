# Frontend para Detección de Violencia Física Escolar

## Instalación

1. **Instalar dependencias**:
   ```bash
   npm install
   ```

2. **Configurar Firebase**:
   - Ve a la consola de Firebase y crea un proyecto.
   - Habilita Firestore como base de datos.
   - Copia las credenciales de Firebase SDK desde **Configuración del Proyecto > General > Tus aplicaciones**.
   - Pega las credenciales en `src/firebase.js`.

## Compilación

1. **Compilar el frontend**:
   ```bash
   npm run build
   ```

2. **Copiar los archivos compilados al backend**:
   - Copia el contenido de `frontend/build/` a `backend/static/`.
   - Por ejemplo, en Linux/Mac:
     ```bash
     cp -r build/* ../backend/static/
     ```

## Ejecución

- El frontend se sirve desde el backend.
- Inicia el servidor backend (`python app.py`) y accede a `http://localhost:5000` en tu navegador.

## Notas
- Asegúrate de que el backend esté ejecutándose para que el frontend pueda conectarse vía WebSocket.