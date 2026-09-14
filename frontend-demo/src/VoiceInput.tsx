import { useDeepgram } from "./useDeepgram";
import "./VoiceInput.css";

interface VoiceInputProps {
  apiKey: string;
  onTranscript: (text: string) => void;
  disabled?: boolean;
}

export function VoiceInput({ apiKey, onTranscript, disabled }: VoiceInputProps) {
  const {
    isRecording,
    transcript,
    finalTranscript,
    startRecording,
    stopRecording,
    error,
  } = useDeepgram(apiKey);

  const handleClick = () => {
    if (isRecording) {
      stopRecording();
      if (finalTranscript.trim()) {
        onTranscript(finalTranscript.trim());
      }
    } else {
      startRecording();
    }
  };

  return (
    <div className="voice-input">
      <button
        type="button"
        className={`voice-input-button${isRecording ? " voice-input-recording" : ""}`}
        onClick={handleClick}
        disabled={disabled}
        aria-label={isRecording ? "Stop recording" : "Start voice input"}
        aria-pressed={isRecording}
      >
        🎤
      </button>

      {isRecording && transcript && (
        <div className="voice-input-interim">{transcript}</div>
      )}

      {error && <div className="voice-input-error">{error}</div>}
    </div>
  );
}
