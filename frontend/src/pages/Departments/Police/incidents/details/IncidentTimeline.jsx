import React from "react";
import Panel from "../components/Panel";
import { formatClock, formatDateTime } from "../utils";

const tone = (label) => {
  const l = label.toLowerCase();
  if (l.includes("escalat")) return "text-orange-400";
  if (l.includes("dispatch")) return "text-blue-400";
  if (l.includes("arriv")) return "text-emerald-400";
  if (l.includes("coordinat") || l.includes("notif")) return "text-amber-400";
  if (l.includes("resolv") || l.includes("clos")) return "text-emerald-400";
  return "text-slate-300";
};

export default function IncidentTimeline({ incident }) {
  // Real events only. If the record has no "reported" event, derive one from
  // the real reported_at field (never invented).
  const events = incident.events.slice();
  const hasReported = events.some((e) => /report/i.test(e.label));
  if (!hasReported && incident.reportedAt) {
    events.unshift({
      id: `${incident.id}-reported`,
      ts: incident.reportedAt,
      label: "Incident Reported",
      description: null,
      unit: null,
      department: null,
    });
  }

  return (
    <Panel title="Incident Timeline" right={<span className="text-[10px] font-mono text-slate-500">{events.length} events</span>}>
      <div className="p-4 max-h-[520px] overflow-y-auto custom-scrollbar">
        {events.length === 0 ? (
          <div className="text-xs text-slate-500">No timeline events recorded.</div>
        ) : (
          <ol className="relative border-l border-slate-700 ml-1.5 space-y-5">
            {events.map((e, idx) => {
              const latest = idx === events.length - 1;
              return (
                <li key={e.id} className={`relative pl-5 ${latest ? "bg-slate-800/60 border border-slate-700 rounded py-2 pr-3 -ml-px" : ""}`}>
                  <span className={`absolute top-3 -left-[7px] rounded-full border-2 border-slate-900 ${latest ? "w-3.5 h-3.5 bg-blue-500 animate-pulse" : "w-3 h-3 bg-slate-600"}`} />
                  <div className="text-[11px] font-mono text-slate-400" title={formatDateTime(e.ts)}>{formatClock(e.ts)}</div>
                  <div className={`text-[11px] font-bold uppercase tracking-wider ${tone(e.label)}`}>{e.label}</div>
                  {e.description && <div className="text-xs text-slate-300 mt-0.5">{e.description}</div>}
                  {(e.unit || e.department) && (
                    <div className="text-[10px] text-slate-500 mt-0.5 font-mono">{[e.department, e.unit].filter(Boolean).join(" · ")}</div>
                  )}
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </Panel>
  );
}
