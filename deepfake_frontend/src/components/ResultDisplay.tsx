// D:\FLOW\CAPSTON\FINAL_SCANNER\deepfake_frontend\src\components\ResultDisplay.tsx
type ResultDisplayProps = {
  result: any;
};

export default function ResultDisplay({ result }: ResultDisplayProps) {
  return (
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
                <span className="font-semibold">Overall Fake Probability:</span>{" "}
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
  );
}
