import { SEVERITY_ORDER, TYPE_FILTERS } from "./constants";

// ---------- primitives ----------
export const norm = (v) => String(v || "").trim().toUpperCase();

export const titleCase = (value) =>
  String(value || "")
    .replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim()
    .split(" ")
    .map((t) => (t && t === t.toUpperCase() && t.length <= 3 ? t : t.charAt(0).toUpperCase() + t.slice(1).toLowerCase()))
    .join(" ");

export const parseJson = (value, fallback) => {
  if (value === null || value === undefined) return fallback;
  if (typeof value === "string") {
    try { return JSON.parse(value); } catch { return fallback; }
  }
  return value;
};

export const toDate = (value) => {
  if (!value) return null;
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? null : d;
};

export const formatClock = (value) => {
  const d = value instanceof Date ? value : toDate(value);
  return d ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false }) : "--:--";
};

export const formatDateTime = (value) => {
  const d = value instanceof Date ? value : toDate(value);
  return d ? d.toLocaleString([], { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", hour12: false }) : "--";
};

export const incidentPath = (id) => `/police/incidents?id=${encodeURIComponent(id)}`;

// ---------- departments / timeline ----------
export const parseDepartments = (value) => {
  const parsed = parseJson(value, []);
  const list = Array.isArray(parsed) ? parsed : [];
  return list
    .map((d) =>
      typeof d === "string"
        ? { name: titleCase(d), role: null, unit: null }
        : { name: titleCase(d?.department || d?.name || d?.dept || ""), role: d?.role ? titleCase(d.role) : null, unit: d?.unit || d?.assigned_unit || null },
    )
    .filter((d) => d.name);
};

export const getPoliceDepartment = (departments) => departments.find((d) => /police/i.test(d.name));

const parseEventTime = (value, fallbackDate) => {
  if (!value) return null;
  const direct = new Date(value);
  if (!Number.isNaN(direct.getTime())) return direct;
  const m = String(value).match(/^(\d{1,2}):(\d{2})(?::(\d{2}))?/);
  if (m) {
    const base = toDate(fallbackDate) || new Date();
    base.setHours(Number(m[1]), Number(m[2]), Number(m[3] || 0), 0);
    return base;
  }
  return null;
};

const normalizeEvents = (timelineRaw, reportedAt, incidentId) => {
  const list = Array.isArray(timelineRaw?.timeline) ? timelineRaw.timeline : Array.isArray(timelineRaw) ? timelineRaw : [];
  const events = list.map((e, i) => ({
    id: `${incidentId}-${i}`,
    ts: parseEventTime(e?.time || e?.timestamp, reportedAt) || reportedAt,
    label: titleCase(e?.event || e?.type || e?.action || e?.title || "Update"),
    description: e?.description || e?.details || e?.message || e?.note || null,
    unit: e?.unit || e?.assigned_unit || null,
    department: e?.department || e?.dept || null,
  }));
  events.sort((a, b) => (a.ts?.getTime() || 0) - (b.ts?.getTime() || 0));
  return events;
};

const matchType = (type) => {
  const s = String(type || "").toLowerCase().replace(/[_-]+/g, " ");
  return TYPE_FILTERS.find((t) => t.keywords.some((k) => s.includes(k)))?.value || "other";
};

// ---------- normalisation ----------
export function normalizeIncident(raw) {
  const reportedAt = toDate(raw.reported_at || raw.created_at);
  const timelineRaw = parseJson(raw.timeline_events, null);
  const departments = parseDepartments(raw.assigned_departments);
  const events = normalizeEvents(timelineRaw, reportedAt, raw.incident_id);
  const incident = {
    ...raw,
    id: String(raw.incident_id),
    reportedAt,
    severityKey: norm(raw.severity) || "UNKNOWN",
    statusKey: norm(raw.status) || "UNKNOWN",
    typeLabel: titleCase(raw.incident_type) || "Incident",
    typeKey: matchType(raw.incident_type),
    location: raw.location || timelineRaw?.location || "Unknown location",
    departments,
    events,
    lastEvent: events[events.length - 1] || null,
  };
  incident.searchText = [incident.id, incident.typeLabel, incident.location, incident.severityKey, incident.statusKey, ...departments.map((d) => d.name)]
    .join(" ").toLowerCase();
  return incident;
}

// ---------- KPIs / filter / sort ----------
export function computeKpis(incidents) {
  const active = incidents.filter((i) => i.statusKey === "ACTIVE");
  const count = (sev) => active.filter((i) => i.severityKey === sev).length;
  return {
    total: incidents.length,
    active: active.length,
    critical: count("CRITICAL"), // severity KPIs describe currently active incidents
    high: count("HIGH"),
    medium: count("MEDIUM"),
    resolved: incidents.filter((i) => i.statusKey === "RESOLVED" || i.statusKey === "CLOSED").length,
  };
}

export function filterIncidents(list, f, nowMs = Date.now()) {
  const q = f.search.trim().toLowerCase();
  const now = new Date(nowMs);
  const from = f.date === "custom" && f.from ? new Date(`${f.from}T00:00:00`) : null;
  const to = f.date === "custom" && f.to ? new Date(`${f.to}T23:59:59.999`) : null;

  return list.filter((i) => {
    if (q && !i.searchText.includes(q)) return false;
    if (f.severity !== "all" && i.severityKey !== f.severity.toUpperCase()) return false;
    if (f.status !== "all" && i.statusKey !== f.status.toUpperCase()) return false;
    if (f.type !== "all" && i.typeKey !== f.type) return false;
    if (f.department !== "all" && !i.departments.some((d) => d.name.toLowerCase().includes(f.department))) return false;

    if (f.date !== "all") {
      const t = i.reportedAt;
      if (!t) return false;
      if (f.date === "today" && t.toDateString() !== now.toDateString()) return false;
      if (f.date === "24h" && nowMs - t.getTime() > 24 * 3600 * 1000) return false;
      if (f.date === "7d" && nowMs - t.getTime() > 7 * 24 * 3600 * 1000) return false;
      if (from && t < from) return false;
      if (to && t > to) return false;
    }
    return true;
  });
}

const byNewest = (a, b) => (b.reportedAt?.getTime() || 0) - (a.reportedAt?.getTime() || 0);
const sevRank = (i) => SEVERITY_ORDER[i.severityKey] ?? 9;

export function sortIncidents(list, { key, dir }) {
  const m = dir === "desc" ? -1 : 1;
  const text = (x, y) => String(x).localeCompare(String(y));
  const comparers = {
    severity: (a, b) => sevRank(a) - sevRank(b),
    time: (a, b) => byNewest(a, b), // "asc" = newest first
    status: (a, b) => text(a.statusKey, b.statusKey),
    type: (a, b) => text(a.typeLabel, b.typeLabel),
    location: (a, b) => text(a.location, b.location),
  };
  const primary = comparers[key] || comparers.severity;
  return list.slice().sort((a, b) => {
    const r = primary(a, b) * m;
    if (r !== 0) return r;
    // Tie-breaks: severity, then newest first.
    return sevRank(a) - sevRank(b) || byNewest(a, b);
  });
}
