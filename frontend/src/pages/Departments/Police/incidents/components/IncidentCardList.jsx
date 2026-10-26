import React from "react";
import { SeverityBadge, StatusBadge, departmentsLabel, severityStyle } from "./badges";
import { formatClock } from "../utils";

export default function IncidentCardList({ rows, onOpen, onPreview }) {
  return (
    <div className="lg:hidden grid grid-cols-1 sm:grid-cols-2 gap-3">
      {rows.map((i) => (
        <div
          key={i.id}
          role="button"
          tabIndex={0}
          onClick={() => onOpen(i.id)}
          onKeyDown={(e) => e.key === "Enter" && onOpen(i.id)}
          className="relative bg-slate-900 border border-slate-700 rounded-lg p-3 pl-4 cursor-pointer hover:border-slate-500 focus:outline-none focus:ring-1 focus:ring-blue-500 overflow-hidden"
        >
          <span className={`absolute left-0 top-0 bottom-0 w-1 ${severityStyle(i.severityKey).accent}`} />

          <div className="flex justify-between items-start gap-2 mb-1">
            <div className="text-sm font-bold uppercase text-slate-100 leading-tight">{i.typeLabel}</div>
            <SeverityBadge severity={i.severityKey} />
          </div>
          <div className="text-[10px] font-mono text-blue-400 truncate mb-1">{i.id}</div>
          <div className="text-xs text-slate-300 mb-2 truncate">{i.location}</div>

          <div className="flex items-center justify-between gap-2 mb-2">
            <StatusBadge status={i.statusKey} />
            <span className="text-[11px] font-mono text-slate-400">Reported {formatClock(i.reportedAt)}</span>
          </div>

          <div className="text-[11px] text-slate-400 truncate mb-2">{departmentsLabel(i.departments)}</div>

          <div className="flex justify-between items-center">
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); onPreview(i.id); }}
              className="text-[10px] uppercase tracking-wider px-2 py-1 rounded border border-slate-600 text-slate-300 hover:bg-slate-700"
            >
              Preview
            </button>
            <span className="text-[11px] font-bold uppercase tracking-wider text-blue-400">View Incident →</span>
          </div>
        </div>
      ))}
    </div>
  );
}
