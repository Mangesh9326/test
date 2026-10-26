import { useEffect } from "react";
import { X } from "lucide-react";
import CameraFeedCard from "./CameraFeedCard";

export default function FullscreenCameraModal({ camera, onClose, agentTick = 0, timeAgo = "just now" }) {
  useEffect(() => {
    const closeOnEscape = (event) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  if (!camera) return null;
  return (
    <div
      className="traffic-camera-fade absolute inset-0 z-[100] flex items-center justify-center bg-slate-950/80 p-4 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label={`${camera.camera_code} fullscreen feed`}
      onMouseDown={onClose}
    >
      <section
        onMouseDown={(event) => event.stopPropagation()}
        className="traffic-camera-modal flex max-h-[calc(100vh-2rem)] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-slate-700 bg-slate-900 shadow-2xl"
      >
        <header className="flex items-start justify-between gap-4 border-b border-slate-700/70 bg-slate-800/70 px-4 py-3 sm:px-5">
          <div>
            <div className="flex items-center gap-3">
              <p className="font-mono text-xs font-bold text-blue-300">
                {camera.camera_code}
              </p>
              {camera.status === "Online" && (
                <span className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-widest text-blue-400 bg-blue-950/40 border border-blue-500/30 px-1.5 py-0.5 rounded">
                   <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" /> Traffic Agent Active &bull; AI Monitoring
                </span>
              )}
            </div>
            <h2 className="mt-1 text-lg font-semibold text-white">
              {camera.name}
            </h2>
            <p className="mt-1 text-xs text-slate-400">
              {camera.direction} · {camera.road_name}
            </p>
          </div>
          <button
            onClick={onClose}
            aria-label="Close fullscreen feed"
            className="rounded-lg border border-slate-600 p-2 text-slate-300 transition hover:bg-slate-700 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
          >
            <X className="h-5 w-5" />
          </button>
        </header>
        <div className="aspect-video min-h-0 bg-black">
          <CameraFeedCard 
            camera={camera} 
            priority 
            agentTick={agentTick} 
            timeAgo={timeAgo} 
          />
        </div>
      </section>
    </div>
  );
}