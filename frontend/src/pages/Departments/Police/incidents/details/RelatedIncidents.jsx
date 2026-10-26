import React from "react";
import { useNavigate } from "react-router-dom";
import Panel from "../components/Panel";
import { SeverityBadge, StatusBadge } from "../components/badges";
import { formatDateTime, incidentPath } from "../utils";

// Reasons are computed from real fields so the list explains why it is related.
const reasonsFor = (current, other) => {
  const r = [];
  if (other.incident_type === current.incident_type) r.push("Same type");
  if (other.location && other.location === current.location) r.push("Same location");
  if (other.distance_km != null && Number(other.distance_km) <= 2) r.push(`${Number(other.distance_km).toFixed(1)} km away`);
  const shared = other.departments.some((d) => current.departments.some((c) => c.name === d.name));
  if (shared && current.departments.length > 1) r.push("Shared departments");
  return r;
};

export default function RelatedIncidents({ current, items }) {
  const navigate = useNavigate();

  return (
    <Panel title="Related Incidents" right={<span className="text-[10px] font-mono text-slate-500">{items.length}</span>}>
      {items.length === 0 ? (
        <div className="p-4 text-xs text-slate-500">No related incidents found.</div>
      ) : (
        <ul className="divide-y divide-slate-800">
          {items.map((i) => (
            <li key={i.id}>
              <button
                type="button"
                onClick={() => navigate(incidentPath(i.id))}
                className="w-full text-left px-4 py-2.5 flex flex-col sm:flex-row sm:items-center gap-2 hover:bg-slate-800/70"
              >
                <div className="sm:w-1/3 min-w-0">
                  <div className="text-xs font-bold text-slate-100 truncate">{i.typeLabel}</div>
                  <div className="text-[10px] font-mono text-blue-400 truncate">{i.id}</div>
                </div>
                <div className="flex items-center gap-2 sm:w-1/3">
                  <SeverityBadge severity={i.severityKey} />
                  <StatusBadge status={i.statusKey} />
                </div>
                <div className="sm:w-1/3 sm:text-right min-w-0">
                  <div className="text-[11px] font-mono text-slate-400">{formatDateTime(i.reportedAt)}</div>
                  <div className="text-[10px] text-slate-500 truncate">{reasonsFor(current, i).join(" · ")}</div>
                </div>
              </button>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}
