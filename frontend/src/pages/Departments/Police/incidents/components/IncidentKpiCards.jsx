import React from "react";

const CARDS = [
  { key: "total", label: "Total", tone: "text-slate-100 border-slate-700 bg-slate-900" },
  { key: "active", label: "Active", tone: "text-blue-400 border-blue-500/30 bg-blue-500/5" },
  { key: "critical", label: "Critical", sub: "of active", tone: "text-red-400 border-red-500/30 bg-red-500/5" },
  { key: "high", label: "High", sub: "of active", tone: "text-orange-400 border-orange-500/30 bg-orange-500/5" },
  { key: "medium", label: "Medium", sub: "of active", tone: "text-amber-400 border-amber-500/30 bg-amber-500/5" },
  { key: "resolved", label: "Resolved", sub: "incl. closed", tone: "text-emerald-400 border-emerald-500/30 bg-emerald-500/5" },
];

export default function IncidentKpiCards({ kpis }) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-6 gap-3">
      {CARDS.map((c) => (
        <div key={c.key} className={`px-3 py-3 rounded-lg border ${c.tone}`}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400">{c.label}</span>
            {c.key === "critical" && kpis.critical > 0 && <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />}
          </div>
          <div className="text-2xl sm:text-3xl font-mono font-bold leading-none">{kpis[c.key]}</div>
          {c.sub && <div className="text-[9px] text-slate-500 mt-1 uppercase tracking-wider">{c.sub}</div>}
        </div>
      ))}
    </div>
  );
}
