import { useCallback, useRef, useState } from "react";
import {
  createClient,
  LiveTranscriptionEvents,
  type LiveClient,
} from "@deepgram/sdk";

interface UseDeepgramResult {
  isRecording: boolean;
  transcript: string;
  finalTranscript: string;
  startRecording: () => void;
  stopRecording: () => void;
  error: string | null;
}

export function useDeepgram(apiKey: string): UseDeepgramResult {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [finalTranscript, setFinalTranscript] = useState("");
  const [error, setError] = useState<string | null>(null);

  const connectionRef = useRef<LiveClient | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const accumulatedTranscriptRef = useRef("");
  const keepAliveIntervalRef = useRef<ReturnType<typeof setInterval> | null>(
    null,
  );

  const cleanup = useCallback(() => {
    if (keepAliveIntervalRef.current) {
      clearInterval(keepAliveIntervalRef.current);
      keepAliveIntervalRef.current = null;
    }

    if (mediaRecorderRef.current) {
      try {
        if (mediaRecorderRef.current.state !== "inactive") {
          mediaRecorderRef.current.stop();
        }
      } catch {
        // ignore
      }
      mediaRecorderRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    if (connectionRef.current) {
      try {
        connectionRef.current.requestClose();
      } catch {
        // ignore
      }
      connectionRef.current = null;
    }

    setIsRecording(false);
  }, []);

  const startRecording = useCallback(() => {
    if (isRecording) return;

    if (!apiKey) {
      setError("Voice input is not configured.");
      return;
    }

    setError(null);
    setTranscript("");
    setFinalTranscript("");
    accumulatedTranscriptRef.current = "";

    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        mediaStreamRef.current = stream;

        const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? "audio/webm;codecs=opus"
          : "audio/webm";

        const deepgram = createClient(apiKey);
        const connection = deepgram.listen.live({
          language: "en-IN",
          model: "nova-2",
          smart_format: true,
          interim_results: true,
        });
        connectionRef.current = connection;

        connection.on(LiveTranscriptionEvents.Open, () => {
          console.log("Deepgram connection open");

          const mediaRecorder = new MediaRecorder(stream, { mimeType });
          mediaRecorderRef.current = mediaRecorder;

          mediaRecorder.addEventListener("dataavailable", (event) => {
            if (event.data.size > 0 && connectionRef.current) {
              connectionRef.current.send(event.data);
            }
          });

          mediaRecorder.start(250);
          setIsRecording(true);

          keepAliveIntervalRef.current = setInterval(() => {
            connectionRef.current?.keepAlive();
          }, 8000);
        });

        connection.on(LiveTranscriptionEvents.Transcript, (data) => {
          const text: string | undefined =
            data?.channel?.alternatives?.[0]?.transcript;

          console.log("Deepgram event:", data.is_final, text);

          if (!text) return;

          if (data.is_final) {
            accumulatedTranscriptRef.current = (
              accumulatedTranscriptRef.current + " " + text
            ).trim();
            setFinalTranscript(accumulatedTranscriptRef.current);
            setTranscript("");
          } else {
            setTranscript(text);
          }
        });

        connection.on(LiveTranscriptionEvents.Error, (err) => {
          console.error("Deepgram error:", err);
          setError("Voice transcription failed. Please try again.");
          cleanup();
        });

        connection.on(LiveTranscriptionEvents.Close, (event) => {
          console.log(
            "Deepgram connection closed:",
            event?.code,
            event?.reason,
          );
          setIsRecording(false);
        });
      })
      .catch(() => {
        setError("Microphone permission denied.");
        setIsRecording(false);
      });
  }, [apiKey, cleanup, isRecording]);

  const stopRecording = useCallback(() => {
    cleanup();
  }, [cleanup]);

  return {
    isRecording,
    transcript,
    finalTranscript,
    startRecording,
    stopRecording,
    error,
  };
}
