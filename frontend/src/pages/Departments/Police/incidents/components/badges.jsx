import React from "react";
import { FALLBACK_SEVERITY, FALLBACK_STATUS, SEVERITY_STYLES, STATUS_STYLES } from "../constants";

export const severityStyle = (key) => SEVERITY_STYLES[key] || FALLBACK_SEVERITY;

// Severity = "how dangerous is this?"  (filled badge, P-rank)
export const SeverityBadge = ({ severity }) => (
  <span className={`inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded border font-mono font-bold uppercase tracking-wider ${severityStyle(severity).badge}`}>
    {severity === "CRITICAL" && <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />}
    {severity || "UNKNOWN"}
  </span>
);

// Status = "what is happening with it?"  (outlined pill with dot)
export const StatusBadge = ({ status }) => {
  const s = STATUS_STYLES[status] || FALLBACK_STATUS;
  return (
    <span className={`inline-flex items-center gap-1.5 text-[10px] px-2 py-0.5 rounded-full border font-mono uppercase tracking-wider ${s.badge}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${s.dot}`} />
      {status || "UNKNOWN"}
    </span>
  );
};

export const PriorityMark = ({ severity }) => {
  const s = severityStyle(severity);
  return (
    <div className="flex items-center gap-2">
      <span className={`w-1 h-8 rounded-sm ${s.accent}`} />
      <span className={`font-mono text-xs font-bold ${s.text}`}>{s.rank}</span>
    </div>
  );
};

export const departmentsLabel = (departments) =>
  departments.length ? departments.map((d) => d.name).join(" + ") : "Police";
