import { Camera, ChevronRight, MapPin } from "lucide-react";

export default function CameraList({ cameras, selectedCameraId, onSelect }) {
  if (!cameras.length)
    return (
      <div className="flex flex-1 flex-col items-center justify-center px-6 text-center text-sm text-slate-500">
        <Camera className="mb-3 h-8 w-8" />
        No cameras match these filters.
      </div>
    );
  return cameras.map((camera) => {
    const selected = camera.camera_id === selectedCameraId;
    return (
      <button
        key={camera.camera_id}
        onClick={() => onSelect(camera.camera_id)}
        className={`group rounded-xl border p-3 text-left transition duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 ${selected ? "border-blue-500/60 bg-blue-500/10 shadow-lg shadow-blue-950/20" : "border-slate-800 bg-slate-900/60 hover:-translate-y-0.5 hover:border-slate-600 hover:bg-slate-800/80"}`}
      >
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="font-mono text-xs font-bold text-blue-300">
              {camera.camera_code}
            </p>
            <p className="mt-1 truncate text-sm font-semibold text-slate-100">
              {camera.name}
            </p>
          </div>
          <span
            className={`mt-0.5 h-2 w-2 shrink-0 rounded-full ${camera.status === "Online" ? "bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,.8)]" : "bg-rose-400"}`}
          />
        </div>
        <div className="mt-3 flex items-center justify-between text-xs text-slate-400">
          <span className="flex min-w-0 items-center gap-1.5 truncate">
            <MapPin className="h-3.5 w-3.5 shrink-0" />
            {camera.direction}
          </span>
          <ChevronRight
            className={`h-4 w-4 transition ${selected ? "translate-x-0 text-blue-300" : "-translate-x-1 text-slate-600 group-hover:translate-x-0"}`}
          />
        </div>
      </button>
    );
  });
}
