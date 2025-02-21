// D:\FLOW\CAPSTON\FINAL_SCANNER\deepfake_frontend\src\app\page.tsx
import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center space-y-8">
      <h2 className="text-4xl font-bold uppercase tracking-widest text-center text-yellow-300 drop-shadow-lg">
        Select Detection Mode
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <Link href="/detect-image">
          <div className="p-6 bg-white/10 backdrop-blur-md rounded-lg shadow-2xl transform transition-all duration-300 hover:scale-105 hover:shadow-2xl cursor-pointer">
            <h3 className="text-2xl font-bold uppercase mb-4 text-center">
              Image Detection
            </h3>
            <p className="text-center">
              Detect deepfakes in images with advanced algorithms.
            </p>
          </div>
        </Link>
        <Link href="/detect-audio">
          <div className="p-6 bg-white/10 backdrop-blur-md rounded-lg shadow-2xl transform transition-all duration-300 hover:scale-105 hover:shadow-2xl cursor-pointer">
            <h3 className="text-2xl font-bold uppercase mb-4 text-center">
              Audio Detection
            </h3>
            <p className="text-center">
              Analyze audio to uncover synthetic voice manipulations.
            </p>
          </div>
        </Link>
        <Link href="/detect-video">
          <div className="p-6 bg-white/10 backdrop-blur-md rounded-lg shadow-2xl transform transition-all duration-300 hover:scale-105 hover:shadow-2xl cursor-pointer">
            <h3 className="text-2xl font-bold uppercase mb-4 text-center">
              Video Detection
            </h3>
            <p className="text-center">
              Examine videos for deepfake content and manipulation.
            </p>
          </div>
        </Link>
      </div>
    </div>
  );
}
