import React, { useEffect, useRef, useState, useCallback } from 'react';
import { db } from './firebase';
import { collection, onSnapshot } from 'firebase/firestore';
import {
  Alert,
  Snackbar,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  Box,
  Container,
  Button,
} from '@mui/material';
import './App.css';

function App() {
  const videoRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [notification, setNotification] = useState(null);
  const [events, setEvents] = useState([]);
  const wsRef = useRef(null);
  const pcRef = useRef(null);

  const startStream = async () => {
    if (isStreaming) return;

    wsRef.current = new WebSocket('ws://192.168.1.6:8000/ws');
    pcRef.current = new RTCPeerConnection({
      iceServers: [
        { urls: "stun:stun.l.google.com:19302" },
      ],
    });

    // Agregar un transceiver para video
    pcRef.current.addTransceiver('video', { direction: 'recvonly' });

    pcRef.current.ontrack = (event) => {
      console.log("Track recibido en el frontend:", event);
      if (event.streams[0]) {
        videoRef.current.srcObject = event.streams[0];
      }
    };

    pcRef.current.onicecandidate = (event) => {
      if (event.candidate) {
        console.log("Enviando ICE candidate desde el frontend:", event.candidate);
        wsRef.current.send(JSON.stringify({ candidate: event.candidate }));
      }
    };

    pcRef.current.onconnectionstatechange = () => {
      console.log("Estado de conexión WebRTC:", pcRef.current.connectionState);
    };

    wsRef.current.onmessage = async (event) => {
      const data = JSON.parse(event.data);
      console.log("Mensaje recibido del backend:", data);
      if (data.answer) {
        await pcRef.current.setRemoteDescription(new RTCSessionDescription(data.answer));
      } else if (data.candidate) {
        await pcRef.current.addIceCandidate(new RTCIceCandidate(data.candidate));
      } else if (data.violence_detected) {
        setNotification(
          `Violencia detectada a las ${data.violence_detected.timestamp}. IDs: ${data.violence_detected.ids_involved.join(", ")}`
        );
      }
    };

    wsRef.current.onopen = async () => {
      console.log("WebSocket abierto en el frontend");
      const offer = await pcRef.current.createOffer();
      await pcRef.current.setLocalDescription(offer);
      wsRef.current.send(JSON.stringify({ offer }));
    };

    wsRef.current.onerror = (error) => {
      console.error('Error en WebSocket:', error);
    };

    wsRef.current.onclose = () => {
      console.log("WebSocket cerrado en el frontend");
    };

    setIsStreaming(true);
  };

  const startDetection = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ start_detection: true }));
      setIsDetecting(true);
    }
  };

  const stopStream = useCallback(() => {
    if (!isStreaming) return;

    if (videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    if (pcRef.current) {
      pcRef.current.close();
    }
    if (wsRef.current) {
      wsRef.current.close();
    }
    setIsStreaming(false);
    setIsDetecting(false);
  }, [isStreaming]);

  const handleStopDetection = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ stop_detection: true }));
      setIsDetecting(false);
    }
  };

  useEffect(() => {
    const unsubscribe = onSnapshot(collection(db, "violence_events"), (snapshot) => {
      const eventList = snapshot.docs.map((doc) => ({ id: doc.id, ...doc.data() }));
      setEvents(eventList);
    });

    return () => {
      stopStream();
      unsubscribe();
    };
  }, [stopStream]);

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom align="center">
        Detección de Violencia Física Escolar en Tiempo Real
      </Typography>

      <Typography variant="body1" align="center" sx={{ mb: 2 }}>
        Detección: {isDetecting ? "Activa" : "Inactiva"}
      </Typography>

      <Box sx={{ mb: 4, display: "flex", justifyContent: "center" }}>
        <Box
          sx={{
            border: "2px solid #ccc",
            borderRadius: "8px",
            overflow: "hidden",
            width: "1280px",
            height: "720px",
          }}
        >
          <video
            ref={videoRef}
            autoPlay
            playsInline
            style={{ width: "100%", height: "100%" }}
          />
        </Box>
      </Box>

      <Box sx={{ mb: 2, display: "flex", justifyContent: "center", gap: 2 }}>
        <Button
          variant="contained"
          color="primary"
          onClick={startStream}
          disabled={isStreaming}
        >
          Iniciar Stream
        </Button>
        <Button
          variant="contained"
          color="primary"
          onClick={startDetection}
          disabled={!isStreaming || isDetecting}
        >
          Iniciar Detección
        </Button>
        <Button
          variant="contained"
          color="secondary"
          onClick={() => {
            stopStream();
            handleStopDetection();
          }}
          disabled={!isStreaming}
        >
          Detener Todo
        </Button>
      </Box>

      <Typography variant="h5" gutterBottom>
        Historial de Eventos de Violencia
      </Typography>
      <TableContainer component={Paper} sx={{ mb: 4 }}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: "bold" }}>Timestamp</TableCell>
              <TableCell sx={{ fontWeight: "bold" }}>Frames</TableCell>
              <TableCell sx={{ fontWeight: "bold" }}>Probabilidad</TableCell>
              <TableCell sx={{ fontWeight: "bold" }}>IDs Involucrados</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {events.length > 0 ? (
              events.map((event) => (
                <TableRow key={event.id}>
                  <TableCell>{event.timestamp || "N/A"}</TableCell>
                  <TableCell>{event.start_frame && event.end_frame ? `${event.start_frame}-${event.end_frame}` : "N/A"}</TableCell>
                  <TableCell>{event.probability ? event.probability.toFixed(4) : "N/A"}</TableCell>
                  <TableCell>{event.ids_involved?.join(", ") || "N/A"}</TableCell>
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={4} align="center">
                  No hay eventos registrados.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <Snackbar
        open={!!notification}
        autoHideDuration={6000}
        onClose={() => setNotification(null)}
        anchorOrigin={{ vertical: "top", horizontal: "center" }}
      >
        <Alert onClose={() => setNotification(null)} severity="error" sx={{ width: "100%" }}>
          {notification}
        </Alert>
      </Snackbar>
    </Container>
  );
}

export default App;