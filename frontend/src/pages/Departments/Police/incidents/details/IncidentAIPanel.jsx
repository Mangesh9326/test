import React, { useEffect, useState } from "react";
import Panel from "../components/Panel";
import { fetchAgentRecommendation } from "../api";
import { norm } from "../utils";

const PRIORITY_STYLE = {
  CRITICAL: "text-red-300 bg-red-500/10 border-red-500/40",
  HIGH: "text-orange-300 bg-orange-500/10 border-orange-500/40",
  MEDIUM: "text-amber-300 bg-amber-500/10 border-amber-500/40",
  LOW: "text-emerald-300 bg-emerald-500/10 border-emerald-500/40",
};

// ---------------------------------------------------------------
// MOCK / SYNTHETIC: used ONLY when the Police Agent has no real
// recommendation to show. It is derived from the incident's real
// severity and is flagged `simulated: true` (shown as "Simulated").
// ---------------------------------------------------------------
const MOCK_BY_SEVERITY = {
  CRITICAL: {
    assessment: "Incident requires sustained police presence and immediate coordination with all responding departments.",
    action: "Establish and maintain a perimeter and deploy additional patrol units to the scene.",
  },
  HIGH: {
    assessment: "Incident requires continued police presence and coordination with nearby response units.",
    action: "Maintain perimeter and deploy an additional patrol unit.",
  },
  MEDIUM: {
    assessment: "Incident is contained but should remain under active monitoring by the responding unit.",
    action: "Keep the assigned unit on scene and review status at the next update cycle.",
  },
  LOW: {
    assessment: "Low-risk incident. Standard response procedures apply.",
    action: "Continue routine handling and update the record when resolved.",
  },
};

const buildMock = (incident) => {
  const m = MOCK_BY_SEVERITY[incident.severityKey] || MOCK_BY_SEVERITY.MEDIUM;
  const level = MOCK_BY_SEVERITY[incident.severityKey] ? incident.severityKey : "MEDIUM";
  return { simulated: true, risk: level, assessment: m.assessment, action: m.action, priority: level };
};

export default function IncidentAIPanel({ incident }) {
  const [real, setReal] = useState(null);

  useEffect(() => {
    let alive = true;
    fetchAgentRecommendation()
      .then((r) => alive && setReal(r))
      .catch((e) => console.error("[INCIDENT AI] agent fetch failed:", e));
    return () => { alive = false; };
  }, []);

  const hasReal = real?.reason && real?.recommendation;
  const view = hasReal
    ? {
        simulated: false,
        risk: norm(real.severity) || incident.severityKey,
        assessment: real.reason,
        action: real.recommendation,
        priority: norm(real.severity) || incident.severityKey,
      }
    : buildMock(incident);

  return (
    <Panel
      title="AI Incident Assessment"
      className="border-blue-500/30"
      right={view.simulated && <span className="text-[9px] font-mono uppercase tracking-widest text-slate-400 border border-slate-600 px-1.5 py-0.5 rounded">Simulated</span>}
    >
      <div className="p-4 space-y-3">
        <div className="grid grid-cols-2 gap-2">
          {[["Risk", view.risk], ["Suggested Priority", view.priority]].map(([label, value]) => (
            <div key={label} className="bg-slate-950 rounded p-2">
              <div className="text-[9px] uppercase tracking-widest text-slate-500 mb-1">{label}</div>
              <span className={`inline-block text-xs font-mono font-bold px-2 py-0.5 rounded border ${PRIORITY_STYLE[value] || PRIORITY_STYLE.MEDIUM}`}>{value}</span>
            </div>
          ))}
        </div>
        <div>
          <div className="text-[9px] uppercase tracking-widest text-slate-500 mb-1">Assessment</div>
          <p className="text-xs text-slate-200 leading-snug">{view.assessment}</p>
        </div>
        <div>
          <div className="text-[9px] uppercase tracking-widest text-slate-500 mb-1">Recommended Action</div>
          <p className="text-xs text-amber-300 bg-amber-500/10 border border-amber-500/20 rounded p-2 leading-snug">{view.action}</p>
        </div>
        {hasReal && <div className="text-[9px] text-slate-500">Latest Police Agent recommendation</div>}
      </div>
    </Panel>
  );
}
