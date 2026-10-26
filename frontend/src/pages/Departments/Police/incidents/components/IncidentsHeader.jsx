import React from "react";
import { formatDateTime } from "../utils";

export default function IncidentsHeader({ total, lastUpdated, onRefresh, refreshing }) {
  return (
    <header className="flex flex-col sm:flex-row justify-between items-start sm:items-end border-b border-slate-700 pb-4 gap-3">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold uppercase tracking-wide text-slate-100">
          Police Incident Management
        </h1>
        <p className="text-xs text-slate-400 font-mono mt-1">
          Monitor, investigate and coordinate active city incidents
        </p>
      </div>

      <div className="flex items-center gap-4 flex-wrap">
        <div className="text-[10px] text-emerald-400 uppercase tracking-widest font-bold flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          Live
        </div>
        <div className="text-[10px] font-mono text-slate-400">
          <span className="text-slate-200 font-bold">{total}</span> incidents
        </div>
        <div className="text-[10px] font-mono text-slate-500">
          Updated {lastUpdated ? formatDateTime(lastUpdated) : "--"}
        </div>
        <button
          type="button"
          onClick={onRefresh}
          disabled={refreshing}
          className="text-[11px] font-bold uppercase tracking-wider px-3 py-1.5 rounded border border-blue-500/40 text-blue-300 bg-blue-500/10 hover:bg-blue-500/20 disabled:opacity-60 flex items-center gap-1.5"
        >
          <span className={refreshing ? "animate-spin inline-block" : "inline-block"}>⟳</span>
          Refresh
        </button>
      </div>
    </header>
  );
}
