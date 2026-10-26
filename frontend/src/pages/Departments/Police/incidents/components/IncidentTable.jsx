import React from "react";
import { PriorityMark, SeverityBadge, StatusBadge, departmentsLabel, severityStyle } from "./badges";
import { formatClock, formatDateTime } from "../utils";

const COLUMNS = [
  { label: "Priority", sortKey: "severity", className: "w-20" },
  { label: "Incident", sortKey: "type" },
  { label: "Location", sortKey: "location" },
  { label: "Severity", sortKey: "severity" },
  { label: "Status", sortKey: "status" },
  { label: "Departments" },
  { label: "Reported", sortKey: "time" },
  { label: "Actions", className: "text-right" },
];

export default function IncidentTable({ rows, sort, onSort, onOpen, onPreview, selectedId }) {
  return (
    <div className="hidden lg:block bg-slate-900 border border-slate-700 rounded-lg overflow-hidden">
      <table className="w-full text-left table-fixed">
        <colgroup>
          <col className="w-[7%]" /><col className="w-[18%]" /><col className="w-[17%]" /><col className="w-[10%]" />
          <col className="w-[11%]" /><col className="w-[17%]" /><col className="w-[9%]" /><col className="w-[11%]" />
        </colgroup>
        <thead className="bg-slate-950 border-b border-slate-700">
          <tr>
            {COLUMNS.map((c) => {
              const active = c.sortKey && c.sortKey === sort.key;
              return (
                <th key={c.label} className={`px-3 py-2 text-[10px] uppercase tracking-widest font-bold ${c.className || ""} ${active ? "text-blue-300" : "text-slate-500"}`}>
                  {c.sortKey ? (
                    <button type="button" onClick={() => onSort(c.sortKey)} className="uppercase tracking-widest hover:text-slate-200">
                      {c.label}{active ? (sort.dir === "asc" ? " ▲" : " ▼") : ""}
                    </button>
                  ) : c.label}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {rows.map((i) => (
            <tr
              key={i.id}
              tabIndex={0}
              onClick={() => onOpen(i.id)}
              onKeyDown={(e) => e.key === "Enter" && onOpen(i.id)}
              className={`border-b border-slate-800 cursor-pointer hover:bg-slate-800/70 focus:outline-none focus:bg-slate-800 ${severityStyle(i.severityKey).row} ${selectedId === i.id ? "bg-slate-800" : ""}`}
            >
              <td className="px-3 py-2.5"><PriorityMark severity={i.severityKey} /></td>
              <td className="px-3 py-2.5 min-w-0">
                <div className="text-sm font-bold text-slate-100 truncate">{i.typeLabel}</div>
                <div className="text-[10px] font-mono text-blue-400 truncate">{i.id}</div>
              </td>
              <td className="px-3 py-2.5 text-xs text-slate-300 truncate" title={i.location}>{i.location}</td>
              <td className="px-3 py-2.5"><SeverityBadge severity={i.severityKey} /></td>
              <td className="px-3 py-2.5"><StatusBadge status={i.statusKey} /></td>
              <td className="px-3 py-2.5 text-xs text-slate-300 truncate" title={departmentsLabel(i.departments)}>{departmentsLabel(i.departments)}</td>
              <td className="px-3 py-2.5 text-[11px] font-mono text-slate-300" title={formatDateTime(i.reportedAt)}>{formatClock(i.reportedAt)}</td>
              <td className="px-3 py-2.5 text-right whitespace-nowrap">
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); onPreview(i.id); }}
                  className="text-[10px] uppercase tracking-wider px-2 py-1 rounded border border-slate-600 text-slate-300 hover:bg-slate-700 mr-1.5"
                >
                  Preview
                </button>
                <span className="text-[10px] uppercase tracking-wider text-blue-400">View →</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
