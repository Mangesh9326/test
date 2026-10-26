import React from "react";
import Panel from "../components/Panel";
import { CORE_DEPARTMENTS } from "../constants";

export default function DepartmentCoordination({ departments }) {
  const find = (name) => departments.find((d) => d.name.toLowerCase().includes(name.toLowerCase()));
  const others = departments.filter((d) => !CORE_DEPARTMENTS.some((c) => d.name.toLowerCase().includes(c.toLowerCase())));

  return (
    <Panel title="Department Coordination">
      <div className="p-3 grid grid-cols-2 gap-2">
        {CORE_DEPARTMENTS.map((name) => {
          const d = find(name);
          return (
            <div key={name} className={`rounded border px-2.5 py-2 ${d ? "border-blue-500/40 bg-blue-500/5" : "border-slate-800 bg-slate-950"}`}>
              <div className="flex items-center gap-1.5 text-xs font-bold text-slate-200">
                <span className={`w-1.5 h-1.5 rounded-full ${d ? "bg-blue-400" : "bg-slate-700"}`} />
                {name}
              </div>
              <div className="text-[10px] text-slate-500 mt-0.5">{d ? d.role || "Assigned" : "Not involved"}</div>
            </div>
          );
        })}
        {others.map((d) => (
          <div key={d.name} className="rounded border border-blue-500/40 bg-blue-500/5 px-2.5 py-2">
            <div className="text-xs font-bold text-slate-200">{d.name}</div>
            <div className="text-[10px] text-slate-500 mt-0.5">{d.role || "Assigned"}</div>
          </div>
        ))}
      </div>
    </Panel>
  );
}
