import React, { useEffect, useState } from "react";
import io from "socket.io-client";
import { db } from "./firebase";
import { collection, onSnapshot } from "firebase/firestore";

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
} from "@mui/material";
import "./App.css";

// Conectar al backend en RunPod
// 190.210.127.129 : 34677 -> 22
// const socket = io("http://190.210.127.129:5000");
const socket = io("http://192.168.1.6:5000");

function App() {
  const [frame, setFrame] = useState(null);
  const [events, setEvents] = useState([]);
  const [notification, setNotification] = useState(null);
  const [isDetecting, setIsDetecting] = useState(false);

  useEffect(() => {
    // Recibir frames del backend
    socket.on("frame", (frameBytes) => {
      const blob = new Blob([frameBytes], { type: "image/jpeg" });
      const url = URL.createObjectURL(blob);
      setFrame(url);
    });

    // Recibir notificaciones de violencia
    socket.on("violence_detected", (event) => {
      setNotification(
        `Violencia detectada a las ${event.timestamp}. IDs: ${event.ids_involved.join(", ")}`
      );
    });

    // Escuchar eventos de Firestore
    const unsubscribe = onSnapshot(collection(db, "violence_events"), (snapshot) => {
      const eventList = snapshot.docs.map((doc) => ({ id: doc.id, ...doc.data() }));
      setEvents(eventList);
    });

    // Limpiar al desmontar el componente
    return () => {
      socket.off("frame");
      socket.off("violence_detected");
      unsubscribe();
    };
  }, []);

  const handleStartDetection = () => {
    socket.emit("start_detection");
    setIsDetecting(true);
  };

  const handleStopDetection = () => {
    socket.emit("stop_detection");
    setIsDetecting(false);
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom align="center">
        Detección de Violencia Física Escolar en Tiempo Real
      </Typography>

      {/* Estado de la detección */}
      <Typography variant="body1" align="center" sx={{ mb: 2 }}>
        Detección: {isDetecting ? "Activa" : "Inactiva"}
      </Typography>

      {/* Video en tiempo real */}
      <Box sx={{ mb: 4, display: "flex", justifyContent: "center" }}>
        <Box
          sx={{
            border: "2px solid #ccc",
            borderRadius: "8px",
            overflow: "hidden",
            maxWidth: "100%",
            aspectRatio: "16/9",
          }}
        >
          {frame ? (
            <img
              src={frame}
              alt="Video en tiempo real"
              style={{ width: "100%", height: "auto" }}
            />
          ) : (
            <Typography variant="body1" sx={{ p: 2 }}>
              Cargando video...
            </Typography>
          )}
        </Box>
      </Box>

      {/* Botones para iniciar y parar la detección */}
      <Box sx={{ mb: 2, display: "flex", justifyContent: "center", gap: 2 }}>
        <Button
          variant="contained"
          color="primary"
          onClick={handleStartDetection}
          disabled={isDetecting}
        >
          Iniciar Detección
        </Button>
        <Button
          variant="contained"
          color="secondary"
          onClick={handleStopDetection}
          disabled={!isDetecting}
        >
          Parar Detección
        </Button>
      </Box>

      {/* Historial de eventos */}
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

      {/* Notificaciones */}
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