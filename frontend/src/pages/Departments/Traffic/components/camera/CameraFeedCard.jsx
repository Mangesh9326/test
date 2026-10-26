import { memo, useState } from "react";
import { Info, LoaderCircle, Maximize2, VideoOff, X } from "lucide-react";
import { getDeterministicTask } from "./trafficCameraUtils";

const StatusBadge = ({ status }) => (
  <div className="flex items-center gap-1.5">
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-bold uppercase tracking-widest backdrop-blur ${status === "Online" ? "border-rose-400/20 bg-slate-950/65 text-white" : "border-slate-600 bg-slate-950/65 text-slate-300"}`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${status === "Online" ? "bg-rose-500 motion-safe:animate-pulse" : "bg-slate-500"}`}
      />
      {status === "Online" ? "Live" : "Offline"}
    </span>
    {status === "Online" && (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-blue-500/30 bg-slate-950/65 px-2.5 py-1 text-[10px] font-bold uppercase tracking-widest text-blue-300 backdrop-blur">
        <span className="h-1.5 w-1.5 rounded-full bg-blue-500 motion-safe:animate-pulse" />
        AI Monitored
      </span>
    )}
  </div>
);

function CameraFeedCard({ camera, onSelect, onExpand, priority = false, agentTick = 0, timeAgo = "just now" }) {
  const [showInfo, setShowInfo] = useState(false);
  const isOffline = camera.status === "Offline";
  const aiTask = !isOffline ? getDeterministicTask(camera.camera_id, agentTick) : null;

  return (
    <article
      onClick={() => onSelect?.(camera)}
      className="group relative flex h-full w-full cursor-pointer items-center justify-center overflow-hidden bg-slate-950 text-left isolation-auto"
    >
      {isOffline ? (
        <div className="flex flex-col items-center gap-3 text-rose-400">
          <VideoOff className="h-10 w-10 motion-safe:animate-pulse" />
          <span className="font-mono text-xs font-bold tracking-[0.2em]">
            SIGNAL LOST
          </span>
        </div>
      ) : !camera.stream_url ? (
        <div className="flex flex-col items-center gap-3 text-slate-500">
          <LoaderCircle className="h-8 w-8 animate-spin" />
          <span className="text-xs font-medium">Connecting to feed…</span>
        </div>
      ) : camera.isVideo ? (
        <video
          src={camera.stream_url}
          autoPlay
          loop
          muted
          playsInline
          preload={priority ? "auto" : "metadata"}
          className="absolute inset-0 h-full w-full object-cover transition duration-500 group-hover:scale-[1.025]"
        />
      ) : (
        <img
          src={camera.stream_url}
          alt={`Traffic feed for ${camera.name}`}
          loading={priority ? "eager" : "lazy"}
          className="absolute inset-0 h-full w-full object-cover transition duration-500 group-hover:scale-[1.025]"
        />
      )}

      {!isOffline && (
        <div className="pointer-events-none absolute inset-x-0 top-0 h-24 bg-linear-to-b from-slate-950/85 via-slate-950/25 to-transparent" />
      )}
      
      <div className="absolute left-3 top-3 flex flex-col gap-2 z-20">
        <div className="flex items-center gap-2">
          <button
            onClick={(event) => {
              event.stopPropagation();
              setShowInfo((value) => !value);
            }}
            aria-label="Show camera details"
            aria-expanded={showInfo}
            className="rounded-lg border border-white/15 bg-slate-950/65 p-2 text-white backdrop-blur transition hover:bg-blue-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
          >
            <Info className="h-4 w-4" />
          </button>
          <StatusBadge status={camera.status} />
        </div>
      </div>

      {onExpand && (
        <button
          onClick={(event) => {
            event.stopPropagation();
            onExpand(camera);
          }}
          aria-label={`Expand ${camera.camera_code}`}
          className="absolute right-3 top-3 rounded-lg border border-white/15 bg-slate-950/65 p-2 text-white opacity-100 backdrop-blur transition hover:scale-105 hover:bg-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 sm:opacity-0 sm:group-hover:opacity-100 z-20"
        >
          <Maximize2 className="h-4 w-4" />
        </button>
      )}

      {!isOffline && (
        <div className="absolute right-3 bottom-14 flex flex-col items-end gap-1.5 pointer-events-none z-20">
          <div className="bg-slate-950/80 backdrop-blur-md border border-slate-700/50 px-2.5 py-1.5 rounded shadow-lg text-[9px] text-slate-300 uppercase text-right">
            <div className="font-bold text-blue-400">● {aiTask}</div>
            <div className="text-slate-500 mt-0.5 font-mono">Updated {timeAgo}</div>
          </div>
        </div>
      )}

      <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-linear-to-t from-slate-950/90 via-slate-950/30 to-transparent p-4 pt-12">
        <p className="font-mono text-xs font-bold text-blue-300">
          {camera.camera_code}
        </p>
        <h3 className="mt-1 truncate text-sm font-semibold text-white">
          {camera.name}
        </h3>
      </div>

      {showInfo && (
        <div
          onClick={(event) => event.stopPropagation()}
          className="traffic-camera-fade absolute inset-0 z-30 flex flex-col bg-slate-950/95 p-4 backdrop-blur-md"
        >
          <div className="flex items-start justify-between border-b border-slate-800 pb-3">
            <div>
              <p className="font-mono text-xs font-bold text-blue-300">
                {camera.camera_code}
              </p>
              <h3 className="mt-1 text-sm font-semibold text-white">
                Camera details
              </h3>
            </div>
            <button
              onClick={() => setShowInfo(false)}
              aria-label="Close details"
              className="rounded-lg p-1.5 text-slate-400 transition hover:bg-slate-800 hover:text-white"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          <dl className="mt-4 grid gap-4 text-xs">
            <div>
              <dt className="font-semibold uppercase tracking-widest text-slate-500">
                Location & axis
              </dt>
              <dd className="mt-1 text-slate-200">
                {camera.direction} on {camera.road_name}
              </dd>
            </div>
            <div>
              <dt className="font-semibold uppercase tracking-widest text-slate-500">
                Coordinates
              </dt>
              <dd className="mt-1 font-mono text-slate-300">
                {camera.latitude.toFixed(5)}°, {camera.longitude.toFixed(5)}°
              </dd>
            </div>
            <div>
              <dt className="font-semibold uppercase tracking-widest text-slate-500">
                Network source
              </dt>
              <dd className="mt-1 text-slate-200">
                {camera.stream_type} via {camera.source}
              </dd>
            </div>
          </dl>
        </div>
      )}
    </article>
  );
}

export default memo(CameraFeedCard);