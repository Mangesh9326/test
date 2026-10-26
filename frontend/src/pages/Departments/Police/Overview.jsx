import React, { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { usePoliceStore } from "../../../store/usePoliceStore";
import PoliceOperationsMap from "./components/PoliceOperationsMap";
import {
  usePoliceOperationsData,
  deriveMetrics,
  buildAgentView,
  buildCoordinationEvents,
  buildCrimeIntelligence,
  parseDepartments,
  getPoliceDepartment,
  formatClock,
  formatDateTime,
  titleCase,
  norm,
} from "./data/Policeoverviewdata";

// ==========================================================
// STYLE TOKENS
// ==========================================================

const SEVERITY = {
  CRITICAL: {
    text: "text-red-400",
    badge: "text-red-300 bg-red-500/10 border-red-500/40",
    accent: "border-l-red-500",
  },
  HIGH: {
    text: "text-orange-400",
    badge: "text-orange-300 bg-orange-500/10 border-orange-500/40",
    accent: "border-l-orange-500",
  },
  MEDIUM: {
    text: "text-amber-400",
    badge: "text-amber-300 bg-amber-500/10 border-amber-500/40",
    accent: "border-l-amber-500",
  },
  LOW: {
    text: "text-emerald-400",
    badge: "text-emerald-300 bg-emerald-500/10 border-emerald-500/40",
    accent: "border-l-emerald-500",
  },
};

const sev = (severity) =>
  SEVERITY[norm(severity)] || {
    text: "text-slate-400",
    badge: "text-slate-300 bg-slate-500/10 border-slate-500/40",
    accent: "border-l-slate-600",
  };

const CONDITION = {
  NORMAL: { text: "text-emerald-400", bar: "bg-emerald-500" },
  ELEVATED: { text: "text-amber-400", bar: "bg-amber-500" },
  CRITICAL: { text: "text-red-400", bar: "bg-red-500" },
};

const PRIORITY_STYLE = {
  CRITICAL: "text-red-300 bg-red-500/10 border-red-500/40",
  HIGH: "text-orange-300 bg-orange-500/10 border-orange-500/40",
  MEDIUM: "text-amber-300 bg-amber-500/10 border-amber-500/40",
  LOW: "text-emerald-300 bg-emerald-500/10 border-emerald-500/40",
};

// ==========================================================
// SHARED UI
// ==========================================================

const Panel = ({ title, right, className = "", children }) => (
  <div
    className={`bg-slate-900 border border-slate-700 rounded-lg shadow-lg flex flex-col min-h-0 ${className}`}
  >
    <div className="px-4 py-2.5 border-b border-slate-700 flex justify-between items-center shrink-0 gap-2">
      <span className="text-xs font-bold text-slate-300 uppercase tracking-widest">
        {title}
      </span>
      {right}
    </div>
    {children}
  </div>
);

const SeverityBadge = ({ severity }) => (
  <span
    className={`text-[9px] px-1.5 py-0.5 rounded border font-mono uppercase tracking-wider ${sev(severity).badge}`}
  >
    {severity || "UNKNOWN"}
  </span>
);

// ==========================================================
// KPI ROW
// ==========================================================

const KpiCard = ({ label, value, tone, pulse = false }) => (
  <div className={`px-3 py-3 rounded-lg border ${tone}`}>
    <div className="flex items-center justify-between gap-2 mb-1">
      <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 truncate">
        {label}
      </span>
      {pulse && <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse shrink-0" />}
    </div>
    <div className="text-2xl sm:text-3xl font-mono font-bold leading-none">{value}</div>
  </div>
);

// ==========================================================
// SECURITY STATUS
// ==========================================================

const SecurityStatus = ({ security }) => {
  const style = CONDITION[security.condition];

  return (
    <Panel title="City Security Status" className="shrink-0">
      <div className="p-4">
        <div className="flex items-end justify-between mb-3">
          <div>
            <span className="text-[10px] uppercase tracking-widest text-slate-500 block mb-0.5">
              Overall Condition
            </span>
            <span className={`text-xl font-bold font-mono tracking-wider ${style.text}`}>
              {security.condition}
            </span>
          </div>
          {security.condition !== "NORMAL" && (
            <span className={`w-2 h-2 rounded-full animate-pulse mb-2 ${style.bar}`} />
          )}
        </div>

        {/* Threat indicator */}
        <div className="grid grid-cols-3 gap-1 mb-4">
          {["NORMAL", "ELEVATED", "CRITICAL"].map((level) => {
            const on = level === security.condition;
            return (
              <div key={level}>
                <div
                  className={`h-1.5 rounded-sm ${on ? CONDITION[level].bar : "bg-slate-700"}`}
                />
                <div
                  className={`text-[8px] font-mono tracking-widest mt-1 text-center ${
                    on ? CONDITION[level].text : "text-slate-600"
                  }`}
                >
                  {level}
                </div>
              </div>
            );
          })}
        </div>

        <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-[11px]">
          {[
            ["Active Incidents", security.active, "text-slate-100"],
            ["Critical", security.critical, "text-red-400"],
            ["High Severity", security.high, "text-orange-400"],
            ["Multi-Dept", security.multiDept, "text-blue-300"],
            ["Active Perimeters", security.perimeters, "text-slate-100"],
          ].map(([label, value, color]) => (
            <div key={label} className="flex justify-between border-b border-slate-800 pb-1">
              <span className="text-slate-400">{label}</span>
              <span className={`font-mono font-bold ${color}`}>{value}</span>
            </div>
          ))}
        </div>
      </div>
    </Panel>
  );
};

// ==========================================================
// ACTIVE INCIDENTS
// ==========================================================

const ActiveIncidents = ({ incidents, onHover }) => {
  const navigate = useNavigate();

  const go = (id) => navigate(`/police/incidents?id=${encodeURIComponent(id)}`);

  return (
    <Panel
      title="Active Incidents"
      className="flex-1 min-h-[280px] max-h-[460px] lg:max-h-none"
      right={
        incidents.length > 0 && (
          <span className="flex items-center gap-1.5 text-[9px] font-mono text-red-400 uppercase tracking-widest">
            <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
            Live
          </span>
        )
      }
    >
      <div className="overflow-y-auto p-2 space-y-2 custom-scrollbar flex-1 min-h-0">
        {incidents.length === 0 ? (
          <div className="text-xs text-slate-500 text-center p-6">
            No active incidents. All sectors reporting normal.
          </div>
        ) : (
          incidents.map((inc) => {
            const departments = parseDepartments(inc.assigned_departments);
            const police = getPoliceDepartment(departments);
            const policeRole = police?.role || (police ? "Responding" : "Pending");
            const deptLabel = departments.length
              ? departments.map((d) => d.name).join(" + ")
              : "Police";
            const s = sev(inc.severity);

            return (
              <div
                key={inc.incident_id}
                role="button"
                tabIndex={0}
                onClick={() => go(inc.incident_id)}
                onKeyDown={(e) => e.key === "Enter" && go(inc.incident_id)}
                onMouseEnter={() => onHover(inc.incident_id)}
                onMouseLeave={() => onHover(null)}
                className={`p-3 bg-slate-800/50 border border-slate-700 border-l-4 ${s.accent} hover:bg-slate-800 hover:border-slate-500 rounded cursor-pointer transition-colors focus:outline-none focus:ring-1 focus:ring-blue-500`}
              >
                <div className="flex justify-between items-start gap-2 mb-1">
                  <SeverityBadge severity={inc.severity} />
                  <span className="text-[10px] font-mono text-slate-400">
                    {formatClock(inc.reported_at || inc.created_at)}
                  </span>
                </div>

                <div className="text-sm font-bold text-slate-100 leading-tight">
                  {titleCase(inc.incident_type) || "Incident"}
                </div>
                <div className="text-[11px] text-slate-300 truncate mb-1.5">
                  {inc.location || "Unknown location"}
                </div>
                <div className="text-[10px] text-slate-400 truncate mb-2">{deptLabel}</div>

                <div className="flex items-center justify-between text-[10px] text-slate-400 bg-slate-950 px-2 py-1 rounded font-mono">
                  <span>
                    Police: <span className="text-slate-200">{policeRole}</span>
                  </span>
                  <span>
                    Depts: <span className="text-slate-200">{departments.length || 1}</span>
                  </span>
                  <span className="text-blue-400 uppercase">{inc.status || "Active"}</span>
                </div>
              </div>
            );
          })
        )}
      </div>
    </Panel>
  );
};

// ==========================================================
// POLICE AI AGENT
// ==========================================================

const PoliceAgentPanel = ({ view }) => {
  const navigate = useNavigate();

  return (
    <Panel
      title="Police AI Agent"
      className="lg:h-[360px] border-blue-500/30"
      right={
        <div className="flex items-center gap-2">
          {view.simulated && (
            <span className="text-[9px] font-mono uppercase tracking-widest text-slate-400 border border-slate-600 px-1.5 py-0.5 rounded">
              Simulated
            </span>
          )}
          <span className="flex items-center gap-1.5 text-[9px] font-mono uppercase tracking-widest text-blue-400">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
            Analyzing
          </span>
        </div>
      }
    >
      <div className="p-4 flex flex-col gap-3 flex-1 min-h-0 overflow-y-auto custom-scrollbar">
        <div>
          <span className="text-[10px] text-slate-500 uppercase tracking-widest block mb-1">
            Current Assessment
          </span>
          <p className="text-sm text-slate-200 leading-snug">{view.assessment}</p>
        </div>

        <div>
          <span className="text-[10px] text-slate-500 uppercase tracking-widest block mb-1">
            Recommendation
          </span>
          <p className="text-sm text-amber-300 bg-amber-500/10 border border-amber-500/20 rounded p-2 leading-snug">
            {view.recommendation}
          </p>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="bg-slate-950 rounded p-2.5">
            <span className="text-[10px] text-slate-500 uppercase tracking-widest block mb-1">
              Priority
            </span>
            <span
              className={`inline-block text-xs font-mono font-bold px-2 py-0.5 rounded border ${
                PRIORITY_STYLE[view.priority] || PRIORITY_STYLE.MEDIUM
              }`}
            >
              {view.priority}
            </span>
          </div>

          {view.confidence !== null && (
            <div className="bg-slate-950 rounded p-2.5">
              <span className="text-[10px] text-slate-500 uppercase tracking-widest block mb-1">
                Confidence
              </span>
              <div className="flex items-center gap-2">
                <span className="text-blue-400 font-mono font-bold text-sm">
                  {view.confidence}%
                </span>
                <div className="flex-1 h-1.5 bg-slate-800 rounded">
                  <div
                    className="h-1.5 bg-blue-500 rounded"
                    style={{ width: `${Math.min(view.confidence, 100)}%` }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="bg-slate-950 rounded p-2.5">
          <span className="text-[10px] text-slate-500 uppercase tracking-widest block mb-0.5">
            Next Action
          </span>
          <span className="text-xs text-slate-200">{view.nextAction}</span>
        </div>

        <button
          type="button"
          onClick={() => navigate("/police/agent")}
          className="mt-auto text-[11px] font-bold uppercase tracking-wider py-2 rounded border border-blue-500/40 text-blue-300 bg-blue-500/10 hover:bg-blue-500/20 transition-colors"
        >
          View AI Analysis →
        </button>
      </div>
    </Panel>
  );
};

// ==========================================================
// COORDINATION ACTIVITY
// ==========================================================

const eventTone = (label) => {
  const l = String(label).toLowerCase();
  if (l.includes("escalat")) return "text-orange-400";
  if (l.includes("dispatch")) return "text-blue-400";
  if (l.includes("traffic")) return "text-amber-400";
  if (l.includes("multi")) return "text-purple-400";
  if (l.includes("resolv") || l.includes("clos")) return "text-emerald-400";
  return "text-slate-300";
};

const CoordinationActivity = ({ events }) => (
  <Panel title="Coordination Activity" className="lg:h-[360px]">
    <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar p-4">
      <div className="relative border-l border-slate-700 ml-1.5 space-y-4">
        {events.map((event, index) => {
          const latest = index === 0;
          return (
            <div
              key={event.id}
              className={`relative pl-5 ${
                latest ? "bg-slate-800/60 border border-slate-700 rounded py-2 pr-3 -ml-px" : ""
              }`}
            >
              <span
                className={`absolute top-3 -left-[7px] rounded-full border-2 border-slate-900 ${
                  latest ? "w-3.5 h-3.5 bg-blue-500 animate-pulse" : "w-3 h-3 bg-slate-600"
                }`}
              />
              <div className="flex items-center gap-2">
                <span className="text-[11px] font-mono text-slate-400">
                  {formatClock(event.ts)}
                </span>
                {event.synthetic && (
                  <span className="text-[8px] uppercase tracking-widest text-slate-600">
                    simulated
                  </span>
                )}
              </div>
              <div
                className={`text-[11px] font-bold uppercase tracking-wider ${eventTone(event.event)}`}
              >
                {event.event}
              </div>
              <div className={`${latest ? "text-sm" : "text-xs"} text-slate-200 leading-snug`}>
                {event.title}
              </div>
              {event.detail && (
                <div className="text-[10px] text-slate-500 mt-0.5">{event.detail}</div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  </Panel>
);

// ==========================================================
// CRIME INTELLIGENCE
// ==========================================================

const CrimeMetric = ({ label, value, tone }) => (
  <div className="bg-slate-950 border border-slate-800 rounded p-3">
    <div className="text-[9px] text-slate-500 uppercase tracking-wider">{label}</div>
    <div className={`text-2xl font-mono font-bold ${tone}`}>{value}</div>
  </div>
);

const SubHeading = ({ children }) => (
  <h4 className="text-[11px] font-bold text-slate-400 uppercase tracking-widest mb-3">
    {children}
  </h4>
);

const CrimeIntelligence = ({ intel }) => {
  const maxTrend = Math.max(...intel.trend.days.map((d) => d.count), 1);
  const up = intel.trend.delta > 0;

  return (
    <div className="border border-purple-500/40 bg-slate-900/40 rounded-lg p-4">
      <div className="flex items-baseline justify-between border-b border-purple-500/30 pb-3 mb-4 gap-2">
        <h2 className="text-sm font-bold text-purple-400 uppercase tracking-widest">
          Crime Intelligence
        </h2>
        <span className="text-[10px] text-slate-500 font-mono hidden sm:block">
          Mumbai · last 7 days
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
        <CrimeMetric label="Total Records" value={intel.metrics.total} tone="text-purple-300" />
        <CrimeMetric label="Crimes Today" value={intel.metrics.today} tone="text-blue-300" />
        <CrimeMetric label="Last 24 Hours" value={intel.metrics.last24h} tone="text-amber-300" />
        <CrimeMetric
          label="Critical / High"
          value={intel.metrics.highSeverity}
          tone="text-red-400"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* DISTRIBUTION */}
        <div>
          <SubHeading>Crime Distribution</SubHeading>
          <div className="space-y-2.5">
            {intel.distribution.map((row) => (
              <div key={row.category}>
                <div className="flex justify-between text-[11px] text-slate-300 mb-1">
                  <span>{row.category}</span>
                  <span className="font-mono text-slate-400">{row.pct}%</span>
                </div>
                <div className="h-1.5 bg-slate-800 rounded">
                  <div
                    className="h-1.5 bg-purple-500 rounded"
                    style={{ width: `${row.pct}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* LOCATIONS + TREND */}
        <div className="space-y-6">
          <div>
            <SubHeading>High-Risk Locations</SubHeading>
            <div className="space-y-1.5">
              {intel.locations.map((loc, i) => (
                <div
                  key={loc.location}
                  className="flex items-center justify-between bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-[11px]"
                >
                  <span className="text-slate-200">
                    <span className="font-mono text-slate-500 mr-2">{i + 1}</span>
                    {loc.location}
                  </span>
                  <span
                    className={`font-mono text-[9px] tracking-wider ${
                      loc.level === "HIGH"
                        ? "text-red-400"
                        : loc.level === "ELEVATED"
                          ? "text-amber-400"
                          : "text-slate-400"
                    }`}
                  >
                    {loc.level}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div>
            <div className="flex items-baseline justify-between">
              <SubHeading>Crime Trend</SubHeading>
              <span
                className={`text-[10px] font-mono mb-3 ${up ? "text-red-400" : "text-emerald-400"}`}
              >
                {up ? "▲" : "▼"} {Math.abs(intel.trend.delta)}% vs avg
              </span>
            </div>
            <div className="flex items-end gap-1.5 h-20">
              {intel.trend.days.map((d, i) => (
                <div key={`${d.label}-${i}`} className="flex-1 flex flex-col items-center gap-1">
                  <div className="w-full flex-1 flex items-end">
                    <div
                      title={`${d.label}: ${d.count}`}
                      className={`w-full rounded-t ${
                        i === intel.trend.days.length - 1 ? "bg-purple-400" : "bg-purple-600/60"
                      }`}
                      style={{ height: `${Math.max((d.count / maxTrend) * 100, 6)}%` }}
                    />
                  </div>
                  <span className="text-[8px] font-mono text-slate-500">{d.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* RECENT */}
        <div>
          <SubHeading>Recent Crime Records</SubHeading>
          <div className="space-y-1.5 max-h-[420px] overflow-y-auto custom-scrollbar pr-1">
            {intel.recent.map((crime) => (
              <div
                key={crime.id}
                className="bg-slate-950 border border-slate-800 rounded px-2.5 py-2 flex justify-between gap-3"
              >
                <div className="min-w-0">
                  <div className="text-[11px] font-bold text-purple-300">{crime.category}</div>
                  <div className="text-[10px] text-slate-400 truncate">{crime.location}</div>
                  <div className="text-[9px] text-slate-500">{crime.status}</div>
                </div>
                <div className="text-right shrink-0">
                  <div className={`text-[10px] font-mono ${sev(crime.severity).text}`}>
                    {crime.severity}
                  </div>
                  <div className="text-[10px] font-mono text-slate-500">
                    {formatClock(crime.ts)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {intel.includesModelledData && (
        <p className="text-[9px] text-slate-600 mt-4">
          Distribution, locations and trend include a modelled historical baseline while record
          volume is low.
        </p>
      )}
    </div>
  );
};

// ==========================================================
// MAIN DASHBOARD
// ==========================================================

export default function PoliceOverview() {
  const {
    activeIncidents: storeIncidents,
    crimes: storeCrimes,
    isSimulation,
    fetchInitialData,
    subscribeToSockets,
    unsubscribeFromSockets,
  } = usePoliceStore();

  // Single source of truth for the map and every panel.
  const data = usePoliceOperationsData(10000);
  const [hoveredIncidentId, setHoveredIncidentId] = useState(null);

  // Existing store + socket behavior is preserved.
  useEffect(() => {
    try {
      Promise.resolve(fetchInitialData?.()).catch((err) =>
        console.error("[POLICE OVERVIEW] fetchInitialData failed:", err),
      );
    } catch (err) {
      console.error("[POLICE OVERVIEW] fetchInitialData failed:", err);
    }
    subscribeToSockets?.();
    return () => unsubscribeFromSockets?.();
  }, [fetchInitialData, subscribeToSockets, unsubscribeFromSockets]);

  // When sockets push a change into the store (new/resolved incident, new
  // crime), re-pull the unified dataset immediately instead of waiting for
  // the next poll.
  const storeSignature = `${(storeIncidents || [])
    .map((i) => `${i.incident_id}:${i.status}`)
    .join("|")}#${(storeCrimes || []).length}`;
  const firstSignature = useRef(true);

  useEffect(() => {
    if (firstSignature.current) {
      firstSignature.current = false;
      return;
    }
    data.refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storeSignature]);

  const { stations, incidents, crimes, overview, agent, error, lastUpdated } = data;

  const metrics = useMemo(
    () => deriveMetrics({ incidents, crimes, stations, overview, now: lastUpdated }),
    [incidents, crimes, stations, overview, lastUpdated],
  );
  const security = metrics.security;
  const agentView = useMemo(
    () => buildAgentView(agent, incidents, metrics),
    [agent, incidents, metrics],
  );
  const coordination = useMemo(() => buildCoordinationEvents(incidents), [incidents]);
  const crimeIntel = useMemo(
    () => buildCrimeIntelligence(crimes, metrics, lastUpdated),
    [crimes, metrics, lastUpdated],
  );

  const kpis = [
    {
      label: "Active Incidents",
      value: metrics.activeIncidents,
      tone: "text-amber-400 border-amber-500/30 bg-amber-500/5",
    },
    {
      label: "Critical Alerts",
      value: metrics.criticalIncidents,
      tone: "text-red-400 border-red-500/30 bg-red-500/5",
      pulse: metrics.criticalIncidents > 0,
    },
    {
      label: "High Priority",
      value: metrics.highIncidents,
      tone: "text-orange-400 border-orange-500/30 bg-orange-500/5",
    },
    {
      label: "Crimes Today",
      value: metrics.crimesToday,
      tone: "text-purple-400 border-purple-500/30 bg-purple-500/5",
    },
    {
      label: "Units Available",
      value: metrics.unitsAvailable,
      tone: "text-emerald-400 border-emerald-500/30 bg-emerald-500/5",
    },
    {
      label: "Units Deployed",
      value: metrics.unitsDeployed,
      tone: "text-blue-400 border-blue-500/30 bg-blue-500/5",
    },
    {
      label: "Police Stations",
      value: metrics.stations,
      tone: "text-sky-400 border-sky-500/30 bg-sky-500/5",
    },
  ];

  return (
    <div className="p-4 sm:p-6 min-h-screen overflow-y-auto bg-slate-950 font-sans text-slate-100 flex flex-col gap-5">
      {/* HEADER */}
      <header className="flex flex-col sm:flex-row justify-between items-start sm:items-end border-b border-slate-700 pb-4 gap-3">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <h1 className="text-xl sm:text-2xl font-bold text-slate-100 uppercase tracking-wide">
              Police Operations Center
            </h1>
            {isSimulation && (
              <span className="bg-amber-500/10 text-amber-400 border border-amber-500/30 px-2 py-0.5 rounded text-[10px] font-bold tracking-widest uppercase animate-pulse">
                Simulation
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400 font-mono">
            AI-Assisted Law Enforcement &amp; Emergency Coordination
          </p>
        </div>

        <div className="text-left sm:text-right">
          <div className="text-[10px] text-emerald-400 uppercase tracking-widest font-bold mb-1 flex items-center sm:justify-end gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            LIVE
          </div>
          <div className="text-[10px] text-slate-500 font-mono">
            Updated {formatDateTime(lastUpdated)}
          </div>
        </div>
      </header>

      {/* KPI ROW */}
      <div className="grid grid-cols-2 sm:grid-cols-4 xl:grid-cols-7 gap-3">
        {kpis.map((kpi) => (
          <KpiCard key={kpi.label} {...kpi} />
        ))}
      </div>

      {/* MAIN OPERATIONS: map ~2/3, panels ~1/3 */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 lg:items-stretch">
        <div className="lg:col-span-8 h-[420px] sm:h-[520px] lg:h-[620px] bg-slate-900 border border-slate-700 rounded-lg overflow-hidden relative shadow-lg">
          <PoliceOperationsMap
            stations={stations}
            incidents={incidents}
            crimes={crimes}
            error={error}
            highlightedIncidentId={hoveredIncidentId}
          />
        </div>

        <div className="lg:col-span-4 flex flex-col gap-5 lg:h-[620px] min-h-0">
          <SecurityStatus security={security} />
          <ActiveIncidents incidents={incidents} onHover={setHoveredIncidentId} />
        </div>
      </div>

      {/* AI + COORDINATION */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <PoliceAgentPanel view={agentView} />
        <CoordinationActivity events={coordination} />
      </div>

      {/* CRIME INTELLIGENCE */}
      <CrimeIntelligence intel={crimeIntel} />
    </div>
  );
}