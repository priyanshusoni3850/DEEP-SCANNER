// D:\FLOW\CAPSTON\FINAL_SCANNER\deepfake_frontend\src\app\layout.tsx
import "./globals.css";
import { Inter } from "next/font/google";
import Link from "next/link";
import { FaHome } from "react-icons/fa";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Deepfake Detector",
  description: "Deepfake Detection Frontend",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body
        className={`${inter.className} bg-gradient-to-br from-gray-900 to-gray-700 text-white`}
      >
        <header className="sticky top-0 z-50 bg-white/10 backdrop-blur-md shadow-lg p-4 flex items-center justify-between">
          <Link href="/">
            <button
              className="flex items-center gap-2 text-white hover:text-yellow-300 transform transition-all duration-300 hover:scale-110"
              title="Home"
            >
              <FaHome size={28} />
              <span className="hidden sm:inline font-semibold">Home</span>
            </button>
          </Link>
          <h1 className="text-3xl font-bold tracking-widest uppercase">
            Deepfake Detector
          </h1>
          {/* Spacer */}
          <div className="w-8" />
        </header>
        <main className="container mx-auto p-8">{children}</main>
      </body>
    </html>
  );
}
