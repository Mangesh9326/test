import { Grid2X2, Map, Search, SlidersHorizontal, Video } from "lucide-react";

const controlBase = "relative z-10 flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 sm:px-4";

export default function TrafficCameraControls({
  viewMode, setViewMode, feedSource, setFeedSource, searchQuery, setSearchQuery,
  areaFilter, setAreaFilter, statusFilter, setStatusFilter, areas, stats,
}) {
  return (
    <>
      <header className="flex flex-col justify-between gap-4 xl:flex-row xl:items-center">
        <div>
          <p className="mb-1 text-xs font-semibold uppercase tracking-[0.2em] text-blue-400">Traffic operations</p>
          <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">Traffic surveillance</h1>
          <p className="mt-1 text-sm text-slate-400">Monitor live corridors, validate incidents, and focus a camera in one click.</p>
        </div>

        <div className="grid grid-cols-3 gap-2 sm:flex">
          {[["Total", stats.total, "text-slate-100"], ["Online", stats.online, "text-emerald-300"], ["Offline", stats.offline, "text-rose-300"]].map(([label, value, color]) => (
            <div key={label} className="rounded-xl border border-slate-700/80 bg-slate-900/75 px-3 py-2 text-center shadow-sm backdrop-blur">
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-500">{label}</div>
              <div className={`mt-0.5 font-mono text-lg font-bold leading-none ${color}`}>{value}</div>
            </div>
          ))}
        </div>
      </header>

      <section className="rounded-2xl border border-slate-700/60 bg-slate-900/60 p-3 shadow-lg shadow-slate-950/10 backdrop-blur-md">
        <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
          <div className="relative min-w-0 flex-1 xl:max-w-md">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
            <input value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} placeholder="Search camera ID or location"
              className="h-11 w-full rounded-xl border border-slate-700 bg-slate-950/80 pl-10 pr-3 text-sm text-slate-100 outline-none transition placeholder:text-slate-500 focus:border-blue-500 focus:ring-4 focus:ring-blue-500/10" />
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <SlidersHorizontal className="ml-1 hidden h-4 w-4 text-slate-500 sm:block" />
            <select value={areaFilter} onChange={(event) => setAreaFilter(event.target.value)} className="h-11 min-w-36 rounded-xl border border-slate-700 bg-slate-950/80 px-3 text-sm text-slate-300 outline-none transition focus:border-blue-500 custom-scrollbar">
              {areas.map((area) => <option key={area} value={area}>{area === "All" ? "All areas" : area}</option>)}
            </select>
            <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)} className="h-11 min-w-32 rounded-xl border border-slate-700 bg-slate-950/80 px-3 text-sm text-slate-300 outline-none transition focus:border-blue-500">
              <option value="All">All status</option><option value="Online">Online</option><option value="Offline">Offline</option>
            </select>
          </div>

          <div className="flex rounded-xl border border-slate-700 bg-slate-950/70 p-1">
            {[["map", Map, "Map"], ["grid", Grid2X2, "Grid"]].map(([mode, Icon, label]) => (
              <button key={mode} onClick={() => setViewMode(mode)} className={`${controlBase} ${viewMode === mode ? "bg-blue-600 text-white shadow-lg shadow-blue-950/40" : "text-slate-400 hover:bg-slate-800 hover:text-slate-100"}`} aria-pressed={viewMode === mode}>
                <Icon className="h-4 w-4" />{label}
              </button>
            ))}
          </div>

          <div className="flex rounded-xl border border-slate-700 bg-slate-950/70 p-1">
            {[["api", "Snapshots"], ["loop", "Video"]].map(([source, label]) => (
              <button key={source} onClick={() => setFeedSource(source)} className={`${controlBase} ${feedSource === source ? "bg-indigo-600 text-white shadow-lg shadow-indigo-950/40" : "text-slate-400 hover:bg-slate-800 hover:text-slate-100"}`} aria-pressed={feedSource === source}>
                {source === "loop" && <Video className="h-4 w-4" />}{label}
              </button>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
