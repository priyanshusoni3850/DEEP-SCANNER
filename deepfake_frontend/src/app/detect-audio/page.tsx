// D:\FLOW\CAPSTON\FINAL_SCANNER\deepfake_frontend\src\app\detect-audio\page.tsx
import UploadForm from "@/components/UploadForm";

export default function DetectAudio() {
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;
  return (
    <div className="min-h-screen flex flex-col items-center space-y-8">
      <h2 className="text-3xl font-bold uppercase tracking-wider text-center text-cyan-400 drop-shadow-lg">
        Audio Deepfake Detection
      </h2>
      <UploadForm endpoint={`${backendUrl}/detect_audio`} fileType="audio" />
    </div>
  );
}
