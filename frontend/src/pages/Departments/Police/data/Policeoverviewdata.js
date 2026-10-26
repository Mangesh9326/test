// ============================================================
// policeOverviewData.js
// Single source of truth for the Police Operations Center.
//
// - usePoliceOperationsData(): fetches stations, map-data, overview and
//   agent recommendation, polls, and exposes ONE dataset that the map,
//   KPIs, panels and crime section all consume.
// - Derivation helpers (metrics, security condition, coordination events,
//   crime intelligence).
// - Deterministic synthetic enrichment. Synthetic records are ALWAYS flagged
//   `synthetic: true` internally, never written to the database, and fade
//   out automatically as real data grows.
// ============================================================

import { useCallback, useEffect, useRef, useState } from "react";

// ------------------------------------------------------------
// Generic helpers
// ------------------------------------------------------------

export const norm = (value) => String(value || "").trim().toUpperCase();

export const titleCase = (value) =>
  String(value || "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ")
    .map((token) =>
      token && token === token.toUpperCase() && token.length <= 3
        ? token
        : token.charAt(0).toUpperCase() + token.slice(1).toLowerCase(),
    )
    .join(" ");

export const parseJson = (value, fallback) => {
  if (value === null || value === undefined) return fallback;
  if (typeof value === "string") {
    try {
      return JSON.parse(value);
    } catch {
      return fallback;
    }
  }
  return value;
};

export const toNum = (value) => {
  if (value === null || value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};

export const toDate = (value) => {
  if (!value) return null;
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? null : d;
};

export const formatClock = (value) => {
  const d = value instanceof Date ? value : toDate(value);
  if (!d) return "--:--";
  return d.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
};

export const formatDateTime = (value) => {
  const d = value instanceof Date ? value : toDate(value);
  if (!d) return "--";
  return d.toLocaleString([], {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
};

const isSameDay = (a, b) =>
  a.getFullYear() === b.getFullYear() &&
  a.getMonth() === b.getMonth() &&
  a.getDate() === b.getDate();

export const crimeTime = (crime) =>
  toDate(crime?.reported_at || crime?.time || crime?.created_at);

// ------------------------------------------------------------
// Departments / timeline parsing
// ------------------------------------------------------------

export const parseDepartments = (value) => {
  const parsed = parseJson(value, []);
  const list = Array.isArray(parsed) ? parsed : [];

  return list
    .map((d) =>
      typeof d === "string"
        ? { name: titleCase(d), role: null }
        : {
            name: titleCase(d?.department || d?.name || d?.dept || ""),
            role: d?.role ? titleCase(d.role) : null,
          },
    )
    .filter((d) => d.name);
};

export const getPoliceDepartment = (departments) =>
  departments.find((d) => /police/i.test(d.name));

export const getTimeline = (incident) => {
  const raw = parseJson(incident?.timeline_events, null);
  const events = Array.isArray(raw?.timeline)
    ? raw.timeline
    : Array.isArray(raw)
      ? raw
      : [];
  return { location: raw?.location || incident?.location || null, events };
};

const parseEventTime = (value, fallbackDate) => {
  if (!value) return null;

  const direct = new Date(value);
  if (!Number.isNaN(direct.getTime())) return direct;

  const match = String(value).match(/^(\d{1,2}):(\d{2})(?::(\d{2}))?/);
  if (match) {
    const base = toDate(fallbackDate) || new Date();
    base.setHours(Number(match[1]), Number(match[2]), Number(match[3] || 0), 0);
    return base;
  }
  return null;
};

// ------------------------------------------------------------
// Station metadata, KPIs, security condition
// ------------------------------------------------------------

export const getStationMeta = (station) => {
  const meta = parseJson(station?.metadata, {});
  return meta && typeof meta === "object" ? meta : {};
};

export function computeSecurity(incidents = []) {
  const active = incidents.length;
  const critical = incidents.filter((i) => norm(i.severity) === "CRITICAL").length;
  const high = incidents.filter((i) => norm(i.severity) === "HIGH").length;
  const multiDept = incidents.filter(
    (i) => parseDepartments(i.assigned_departments).length > 1,
  ).length;

  // Perimeters are only counted when the incident data actually says so
  // (a perimeter/cordon timeline event or department role).
  const perimeters = incidents.filter((incident) => {
    const { events } = getTimeline(incident);
    const inTimeline = events.some((e) =>
      /perimeter|cordon/i.test(`${e?.event || ""} ${e?.type || ""} ${e?.description || ""}`),
    );
    const inRole = parseDepartments(incident.assigned_departments).some((d) =>
      /perimeter|cordon/i.test(d.role || ""),
    );
    return inTimeline || inRole;
  }).length;

  let condition = "NORMAL";
  if (active >= 3 || high >= 2) condition = "ELEVATED";
  if (active >= 5 || critical > 0) condition = "CRITICAL";

  return { active, critical, high, multiDept, perimeters, condition };
}

export function deriveMetrics({ incidents, crimes, stations, overview, now }) {
  const security = computeSecurity(incidents);
  const today = now || new Date();

  const crimesToday = crimes.filter((c) => {
    const d = crimeTime(c);
    return d && isSameDay(d, today);
  }).length;

  const crimes24h = crimes.filter((c) => {
    const d = crimeTime(c);
    return d && today.getTime() - d.getTime() <= 24 * 3600 * 1000;
  }).length;

  const highSeverityCrimes = crimes.filter((c) =>
    ["CRITICAL", "HIGH"].includes(norm(c.severity)),
  ).length;

  // Units available: sum of patrolsAvailable across real stations
  // (same definition the backend overview uses).
  let available = 0;
  let availableKnown = false;
  let deployed = 0;
  let deployedKnown = false;

  stations.forEach((station) => {
    const meta = getStationMeta(station);
    const avail = toNum(meta.patrolsAvailable);
    const total = toNum(meta.patrolsTotal ?? meta.totalPatrols);
    const out = toNum(meta.patrolsDeployed);

    if (avail !== null) {
      available += avail;
      availableKnown = true;
    }
    if (out !== null) {
      deployed += out;
      deployedKnown = true;
    } else if (total !== null && avail !== null) {
      deployed += Math.max(total - avail, 0);
      deployedKnown = true;
    }
  });

  if (!availableKnown) available = toNum(overview?.unitsAvailable) ?? 0;

  // Fallback that is logically valid: one police response per active
  // incident that has police assigned. (The backend's stations*3 is not used.)
  if (!deployedKnown) {
    deployed = incidents.filter((i) =>
      getPoliceDepartment(parseDepartments(i.assigned_departments)),
    ).length;
  }

  return {
    activeIncidents: security.active,
    criticalIncidents: security.critical,
    highIncidents: security.high,
    crimesToday,
    crimes24h,
    highSeverityCrimes,
    totalCrimes: crimes.length,
    stations: stations.length,
    unitsAvailable: available,
    unitsDeployed: deployed,
    security,
  };
}

// ------------------------------------------------------------
// Police AI agent
// ------------------------------------------------------------

const PRIORITY_BY_CONDITION = { CRITICAL: "CRITICAL", ELEVATED: "HIGH", NORMAL: "LOW" };

export function buildAgentView(realRecommendation, incidents, metrics) {
  if (realRecommendation?.reason && realRecommendation?.recommendation) {
    return {
      simulated: false,
      assessment: realRecommendation.reason,
      recommendation: realRecommendation.recommendation,
      priority: norm(realRecommendation.severity) || "MEDIUM",
      confidence: toNum(realRecommendation.confidence), // only real confidence
      nextAction: realRecommendation.next_action || "Review available patrol units",
      meta: {
        agent: realRecommendation.agent,
        location: realRecommendation.location,
        at: realRecommendation.created_at,
      },
    };
  }

  // Deterministic synthetic advisory, derived from the real incident data.
  const counts = {};
  incidents.forEach((i) => {
    const loc = i.location;
    if (loc && !/unknown|target perimeter/i.test(loc)) counts[loc] = (counts[loc] || 0) + 1;
  });
  const topLocation = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0];
  const n = metrics.activeIncidents;
  const hp = metrics.criticalIncidents + metrics.highIncidents;

  const assessment =
    n > 0
      ? `Elevated police activity detected with ${n} active incident${n > 1 ? "s" : ""}${
          topLocation ? `, concentrated around ${topLocation}` : ""
        }.${hp > 0 ? ` ${hp} rated high priority or above.` : ""} Recent incident density indicates increased patrol demand around major commercial corridors.`
      : "No active incidents reported. Patrol posture is routine and recent crime activity is within the expected range for western Mumbai.";

  const recommendation =
    n > 0
      ? `Increase patrol coverage${
          topLocation ? ` around ${topLocation}` : " in high-density zones"
        } and maintain rapid response availability near high-density zones.`
      : "Maintain standard patrol coverage and keep rapid response units on standby.";

  return {
    simulated: true,
    assessment,
    recommendation,
    priority: PRIORITY_BY_CONDITION[metrics.security.condition] || "MEDIUM",
    confidence: 87,
    nextAction: "Review available patrol units",
    meta: {},
  };
}

// ------------------------------------------------------------
// Coordination activity
// ------------------------------------------------------------

const SYNTHETIC_GENERIC = [
  { event: "TRAFFIC COORDINATION", title: "Western Express Highway", detail: "Traffic department notified" },
  { event: "PATROL UNIT REASSIGNED", title: "Andheri East sector", detail: "Unit P-231 reassigned" },
  { event: "MULTI-DEPARTMENT RESPONSE", title: "Command channel", detail: "Police + Fire + Medical" },
  { event: "POLICE UNIT DISPATCHED", title: "Bandra West", detail: "Unit P-214 assigned" },
  { event: "CHECKPOINT ESTABLISHED", title: "Kurla junction", detail: "Unit P-209 on station" },
  { event: "INCIDENT ESCALATED", title: "Dadar corridor", detail: "Priority changed to HIGH" },
];

export function buildCoordinationEvents(incidents, limit = 8) {
  const real = [];

  incidents.forEach((incident) => {
    const { location, events } = getTimeline(incident);
    const type = titleCase(incident.incident_type);

    events.forEach((e, idx) => {
      const ts =
        parseEventTime(e?.time || e?.timestamp, incident.reported_at) ||
        toDate(incident.reported_at);
      if (!ts) return;

      real.push({
        id: `real-${incident.incident_id}-${idx}`,
        ts,
        event: titleCase(e?.event || e?.type || e?.action || e?.title || "Incident Update"),
        title: location ? `${type} — ${location}` : type,
        detail: e?.unit || e?.department || e?.actor || e?.description || e?.details || null,
        incidentId: incident.incident_id,
        synthetic: false,
      });
    });
  });

  real.sort((a, b) => b.ts - a.ts);
  const merged = real.slice(0, limit);

  if (merged.length < 6) {
    const anchorMs =
      merged[0]?.ts?.getTime() ??
      toDate(incidents[0]?.reported_at)?.getTime() ??
      Math.floor(Date.now() / 300000) * 300000; // stable within 5 min

    const synthetic = [];

    // Incident-linked synthetic events first (derived from real incidents).
    incidents.slice(0, 3).forEach((incident, i) => {
      const { location } = getTimeline(incident);
      const type = titleCase(incident.incident_type);
      const depts = parseDepartments(incident.assigned_departments);
      const base = toDate(incident.reported_at)?.getTime() ?? anchorMs;
      const title = location ? `${type} — ${location}` : type;

      if (!merged.some((e) => e.incidentId === incident.incident_id && /dispatch/i.test(e.event))) {
        synthetic.push({
          event: "POLICE UNIT DISPATCHED",
          title,
          detail: `Unit P-${214 + i * 7} assigned`,
          ts: new Date(base + 60000),
          incidentId: incident.incident_id,
        });
      }
      if (["CRITICAL", "HIGH"].includes(norm(incident.severity))) {
        synthetic.push({
          event: "INCIDENT ESCALATED",
          title,
          detail: `Priority ${norm(incident.severity)}`,
          ts: new Date(base + 120000),
          incidentId: incident.incident_id,
        });
      }
      if (depts.length > 1) {
        synthetic.push({
          event: "MULTI-DEPARTMENT RESPONSE",
          title,
          detail: depts.map((d) => d.name).join(" + "),
          ts: new Date(base + 180000),
          incidentId: incident.incident_id,
        });
      }
    });

    // Generic synthetic coordination, spaced backwards from the anchor.
    SYNTHETIC_GENERIC.forEach((g, i) => {
      synthetic.push({ ...g, ts: new Date(anchorMs - (i + 1) * 4 * 60000), incidentId: null });
    });

    synthetic
      .sort((a, b) => b.ts - a.ts)
      .forEach((s, i) => merged.push({ ...s, id: `syn-${i}`, synthetic: true }));
  }

  return merged.sort((a, b) => b.ts - a.ts).slice(0, limit);
}

// ------------------------------------------------------------
// Crime intelligence (real records + faded synthetic baseline)
// ------------------------------------------------------------

const BASELINE_CATEGORIES = [
  ["Robbery", 16],
  ["Vehicle Theft", 12],
  ["Suspicious Activity", 9],
  ["Burglary", 7],
  ["Assault", 6],
  ["Theft", 5],
  ["Vandalism", 3],
];

const BASELINE_LOCATIONS = ["Bandra West", "Andheri East", "Kurla", "Dadar", "Kurla West"];
const BASELINE_TREND = [14, 11, 16, 13, 18, 15, 12];

const BASELINE_RECENT = [
  { category: "Robbery", location: "Bandra West", severity: "HIGH", status: "Under Investigation" },
  { category: "Vehicle Theft", location: "Andheri East", severity: "MEDIUM", status: "Reported" },
  { category: "Suspicious Activity", location: "Kurla", severity: "LOW", status: "Monitoring" },
  { category: "Burglary", location: "Dadar", severity: "MEDIUM", status: "Under Investigation" },
  { category: "Assault", location: "Kurla West", severity: "HIGH", status: "Under Investigation" },
  { category: "Theft", location: "Bandra West", severity: "LOW", status: "Reported" },
  { category: "Vandalism", location: "Andheri East", severity: "LOW", status: "Closed" },
  { category: "Robbery", location: "Dadar", severity: "HIGH", status: "Under Investigation" },
];

const SEV_WEIGHT = { CRITICAL: 4, HIGH: 3, MEDIUM: 2, LOW: 1 };

export function buildCrimeIntelligence(crimes = [], metrics, now = new Date()) {
  const realTotal = crimes.length;
  // Synthetic baseline fades out as real data grows (gone at 60+ real records).
  const factor = Math.max(0, 1 - realTotal / 60);

  // ---- Distribution
  const counts = {};
  BASELINE_CATEGORIES.forEach(([cat, w]) => {
    counts[cat] = Math.round(w * factor);
  });
  crimes.forEach((c) => {
    const cat = titleCase(c.category || "Unknown");
    counts[cat] = (counts[cat] || 0) + 1;
  });
  const distTotal = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  const distribution = Object.entries(counts)
    .filter(([, count]) => count > 0)
    .map(([category, count]) => ({
      category,
      count,
      pct: Math.round((count / distTotal) * 100),
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 7);

  // ---- High-risk locations
  const locScore = {};
  crimes.forEach((c) => {
    const loc = String(c.location || "").trim();
    if (!loc) return;
    const key = titleCase(loc);
    locScore[key] = (locScore[key] || { reports: 0, score: 0 });
    locScore[key].reports += 1;
    locScore[key].score += SEV_WEIGHT[norm(c.severity)] || 1;
  });
  const locations = Object.entries(locScore)
    .sort((a, b) => b[1].score - a[1].score)
    .map(([location, v]) => ({ location, reports: v.reports, synthetic: false }));
  BASELINE_LOCATIONS.forEach((loc) => {
    if (locations.length < 5 && !locations.some((l) => l.location.toLowerCase() === loc.toLowerCase())) {
      locations.push({ location: loc, reports: null, synthetic: true });
    }
  });
  const rankedLocations = locations.slice(0, 5).map((l, i) => ({
    ...l,
    level: i < 2 ? "HIGH" : i < 4 ? "ELEVATED" : "MODERATE",
  }));

  // ---- Recent records: real first, synthetic fills (older than oldest real)
  const sortedReal = crimes
    .slice()
    .sort((a, b) => (crimeTime(b)?.getTime() || 0) - (crimeTime(a)?.getTime() || 0))
    .slice(0, 8)
    .map((c) => ({
      id: c.id,
      category: titleCase(c.category || "Unknown"),
      location: c.location || "Unknown location",
      severity: norm(c.severity) || "UNKNOWN",
      status: titleCase(c.status || "Reported"),
      ts: crimeTime(c),
      synthetic: false,
    }));

  const recent = sortedReal.slice();
  if (factor > 0 && recent.length < 8) {
    const oldest = recent[recent.length - 1]?.ts?.getTime();
    const anchor = Math.min(oldest ?? Infinity, Math.floor(now.getTime() / 900000) * 900000);
    const offsets = [35, 80, 140, 200, 300, 420, 600, 780];
    BASELINE_RECENT.slice(0, 8 - recent.length).forEach((r, i) => {
      recent.push({
        ...r,
        id: `modelled-${i}`,
        severity: r.severity,
        ts: new Date(anchor - offsets[i] * 60000),
        synthetic: true,
      });
    });
  }

  // ---- 7-day trend
  const days = [];
  for (let i = 6; i >= 0; i -= 1) {
    const d = new Date(now);
    d.setDate(d.getDate() - i);
    const realCount = crimes.filter((c) => {
      const t = crimeTime(c);
      return t && isSameDay(t, d);
    }).length;
    days.push({
      label: d.toLocaleDateString([], { weekday: "short" }),
      count: realCount + Math.round(BASELINE_TREND[6 - i] * factor),
    });
  }
  const prior = days.slice(0, 6).reduce((s, d) => s + d.count, 0) / 6;
  const today = days[6].count;
  const delta = prior > 0 ? Math.round(((today - prior) / prior) * 100) : 0;

  return {
    distribution,
    locations: rankedLocations,
    recent,
    trend: { days, delta },
    includesModelledData: factor > 0,
    metrics: {
      total: metrics.totalCrimes,
      today: metrics.crimesToday,
      last24h: metrics.crimes24h,
      highSeverity: metrics.highSeverityCrimes,
    },
  };
}

// ------------------------------------------------------------
// Data hook: ONE fetch cycle feeding the map and every panel
// ------------------------------------------------------------

const fetchJson = async (url) => {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} returned ${response.status}`);
  return response.json();
};

export function usePoliceOperationsData(pollMs = 10000) {
  const [stations, setStations] = useState([]);
  const [incidents, setIncidents] = useState([]);
  const [crimes, setCrimes] = useState([]);
  const [overview, setOverview] = useState(null);
  const [agent, setAgent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(new Date());

  const mounted = useRef(true);
  const inFlight = useRef(false);

  const refresh = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;

    try {
      const [st, map, ov, ag] = await Promise.allSettled([
        fetchJson("/api/police/stations"),
        fetchJson("/api/police/map-data"),
        fetchJson("/api/police/overview"),
        fetchJson("/api/police/agent/recommendation"),
      ]);

      if (!mounted.current) return;

      // Each source updates independently; a failure keeps the last good data.
      if (st.status === "fulfilled") {
        setStations(Array.isArray(st.value?.stations) ? st.value.stations : []);
      } else console.error("[POLICE OVERVIEW] stations:", st.reason);

      if (map.status === "fulfilled") {
        setIncidents(Array.isArray(map.value?.incidents) ? map.value.incidents : []);
        setCrimes(Array.isArray(map.value?.crimes) ? map.value.crimes : []);
      } else console.error("[POLICE OVERVIEW] map-data:", map.reason);

      if (ov.status === "fulfilled") setOverview(ov.value);
      else console.error("[POLICE OVERVIEW] overview:", ov.reason);

      if (ag.status === "fulfilled") setAgent(ag.value || null);
      else console.error("[POLICE OVERVIEW] agent:", ag.reason);

      const coreFailed = st.status === "rejected" && map.status === "rejected";
      setError(coreFailed ? "Unable to reach police services" : null);
      setLastUpdated(new Date());
    } finally {
      inFlight.current = false;
      if (mounted.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    refresh();
    const interval = setInterval(refresh, pollMs);
    return () => {
      mounted.current = false;
      clearInterval(interval);
    };
  }, [refresh, pollMs]);

  return { stations, incidents, crimes, overview, agent, loading, error, lastUpdated, refresh };
}