"use client";
import { useState, ChangeEvent, FormEvent } from "react";

type UploadFormProps = {
  endpoint: string;
  fileType: string;
};

export default function UploadForm({ endpoint, fileType }: UploadFormProps) {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  // <-- New state to hold status messages from backend
  const [status, setStatus] = useState<string>("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResult(null); // clear previous result when a new file is selected
      setStatus("");
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append(fileType, file);
    try {
      const res = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });
      
      // Only use streaming response for video detection endpoint
      if (endpoint.includes("detect_video")) {
        const reader = res.body?.getReader();
        const decoder = new TextDecoder("utf-8");
        let done = false;
        let buffer = "";
        while (!done) {
          const { value, done: doneReading } = await reader!.read();
          done = doneReading;
          buffer += decoder.decode(value || new Uint8Array(), { stream: !doneReading });
          const lines = buffer.split("\n");
          // Process all complete lines
          for (let i = 0; i < lines.length - 1; i++) {
            const line = lines[i].trim();
            if (line) {
              try {
                const parsed = JSON.parse(line);
                // If the parsed JSON has a status message, update status; otherwise, update result.
                if (parsed.status) {
                  setStatus(parsed.status);
                } else {
                  setResult(parsed);
                }
              } catch (err) {
                console.error("Error parsing line:", line, err);
              }
            }
          }
          // Retain any incomplete line in the buffer.
          buffer = lines[lines.length - 1];
        }
        // Process any remaining buffer after the stream ends.
        if (buffer.trim()) {
          try {
            const parsed = JSON.parse(buffer);
            if (!parsed.status) {
              setResult(parsed);
            }
          } catch (err) {
            console.error("Error parsing final buffer:", buffer, err);
          }
        }
      } else {
        // For non-video endpoints, simply parse the full JSON response.
        const data = await res.json();
        setResult(data);
      }
    } catch (error) {
      console.error("Error:", error);
      setResult({ error: "An error occurred" });
    }
    setLoading(false);
  };

  return (
    <div className="max-w-lg mx-auto p-6 bg-white/10 backdrop-blur-md rounded-lg shadow-2xl">
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <input
          type="file"
          accept={`${fileType}/*`}
          onChange={handleFileChange}
          className="p-3 bg-transparent border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-cyan-400"
        />
        <button
          type="submit"
          className="px-4 py-2 bg-cyan-500 text-white rounded shadow-md hover:bg-cyan-600 transform transition-all duration-300 disabled:opacity-50"
          disabled={loading}
        >
          {loading ? "Processing..." : "Submit"}
        </button>
      </form>
      {fileType === "video" && (
        <p className="text-sm text-gray-300 mt-2">
          Note: Video processing may take up to 30 minutes. Please be patient.
        </p>
      )}
      {/* New status display for streaming heartbeat updates */}
      {status && (
        <div className="mt-4 p-4 bg-gray-800 rounded">
          <p className="text-yellow-300">{status}</p>
        </div>
      )}
      {result && (
        <div className="mt-6 p-6 bg-black/50 rounded-lg shadow-xl">
          {result.error ? (
            <p className="text-red-400 font-semibold">{result.error}</p>
          ) : (
            <>
              <h3 className="text-2xl font-bold mb-4 text-center">
                Detection Result
              </h3>
              <div className="mb-4 space-y-1">
                {result.fake_probability !== undefined && (
                  <p>
                    <span className="font-semibold">Fake Probability:</span>{" "}
                    {result.fake_probability.toFixed(4)}
                  </p>
                )}
                {result.overall_fake_probability !== undefined && (
                  <p>
                    <span className="font-semibold">
                      Overall Fake Probability:
                    </span>{" "}
                    {result.overall_fake_probability.toFixed(4)}
                  </p>
                )}
                {result.label && (
                  <p>
                    <span className="font-semibold">Label:</span> {result.label}
                  </p>
                )}
                {result.final_prediction && (
                  <p>
                    <span className="font-semibold">Final Prediction:</span>{" "}
                    {result.final_prediction}
                  </p>
                )}
                {result.video_verdict && (
                  <p>
                    <span className="font-semibold">Video Verdict:</span>{" "}
                    {result.video_verdict}
                  </p>
                )}
              </div>
              <div className="flex flex-wrap gap-4 justify-center">
                {result.report_url && (
                  <a
                    href={result.report_url}
                    download
                    className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-all duration-300"
                  >
                    Download Report
                  </a>
                )}
                {result.highlighted_image_url && (
                  <a
                    href={result.highlighted_image_url}
                    download
                    className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 transition-all duration-300"
                  >
                    Download Highlighted Image
                  </a>
                )}
                {result.highlighted_video_url && (
                  <a
                    href={result.highlighted_video_url}
                    download
                    className="px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 transition-all duration-300"
                  >
                    Download Highlighted Video
                  </a>
                )}
                {result.probability_plot_url && (
                  <a
                    href={result.probability_plot_url}
                    download
                    className="px-4 py-2 bg-indigo-500 text-white rounded hover:bg-indigo-600 transition-all duration-300"
                  >
                    Download Probability Plot
                  </a>
                )}
                {result.waveform_plot_url && (
                  <a
                    href={result.waveform_plot_url}
                    download
                    className="px-4 py-2 bg-yellow-500 text-white rounded hover:bg-yellow-600 transition-all duration-300"
                  >
                    Download Waveform Plot
                  </a>
                )}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
