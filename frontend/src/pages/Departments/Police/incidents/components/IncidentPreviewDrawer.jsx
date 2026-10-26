import React, { useEffect } from "react";
import { SeverityBadge, StatusBadge, departmentsLabel } from "./badges";
import { formatClock, formatDateTime, getPoliceDepartment } from "../utils";

const Row = ({ label, children }) => (
  <div className="py-2 border-b border-slate-800">
    <div className="text-[9px] uppercase tracking-widest text-slate-500 mb-0.5">{label}</div>
    <div className="text-xs text-slate-200">{children}</div>
  </div>
);

export default function IncidentPreviewDrawer({ incident, onClose, onOpen }) {
  useEffect(() => {
    if (!incident) return undefined;
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [incident, onClose]);

  if (!incident) return null;

  const police = getPoliceDepartment(incident.departments);
  const policeRole = police ? police.role || "Responding" : "Not assigned";
  const last = incident.lastEvent;

  return (
    <div className="fixed inset-0 z-[2000]" role="dialog" aria-modal="true" aria-label="Incident preview">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <aside className="absolute right-0 top-0 bottom-0 w-full sm:w-[400px] bg-slate-900 border-l border-slate-700 shadow-2xl flex flex-col">
        <div className="p-4 border-b border-slate-700">
          <div className="flex items-center gap-2 mb-2">
            <SeverityBadge severity={incident.severityKey} />
            <StatusBadge status={incident.statusKey} />
          </div>
          <h2 className="text-lg font-bold text-slate-100 leading-tight">{incident.typeLabel}</h2>
          <div className="text-[11px] font-mono text-blue-400 break-all">{incident.id}</div>
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar px-4">
          <Row label="Location">{incident.location}</Row>
          <Row label="Reported"><span className="font-mono">{formatDateTime(incident.reportedAt)}</span></Row>
          <Row label="Assigned Departments">{departmentsLabel(incident.departments)}</Row>
          <Row label="Current Police Role">{policeRole}</Row>
          <Row label="Last Timeline Event">
            {last ? (
              <>
                <div className="font-bold uppercase tracking-wide text-[11px]">{last.label}</div>
                <div className="font-mono text-slate-400 text-[10px]">{formatClock(last.ts)}</div>
                {last.description && <div className="text-slate-400 mt-0.5">{last.description}</div>}
              </>
            ) : (
              <span className="text-slate-500">No timeline events recorded</span>
            )}
          </Row>
        </div>

        <div className="p-4 border-t border-slate-700 flex gap-2">
          <button type="button" onClick={() => onOpen(incident.id)} className="flex-1 text-[11px] font-bold uppercase tracking-wider py-2 rounded border border-blue-500/50 text-blue-200 bg-blue-500/15 hover:bg-blue-500/25">
            View Full Incident →
          </button>
          <button type="button" onClick={onClose} className="text-[11px] font-bold uppercase tracking-wider px-4 py-2 rounded border border-slate-600 text-slate-300 hover:bg-slate-800">
            Close
          </button>
        </div>
      </aside>
    </div>
  );
}
