export const PAGE_SIZE = 20;

export const SEVERITY_ORDER = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };

export const SEVERITY_STYLES = {
  CRITICAL: { text: "text-red-400", badge: "text-red-300 bg-red-500/15 border-red-500/50", accent: "bg-red-500", row: "bg-red-500/5", rank: "P1" },
  HIGH: { text: "text-orange-400", badge: "text-orange-300 bg-orange-500/10 border-orange-500/40", accent: "bg-orange-500", row: "", rank: "P2" },
  MEDIUM: { text: "text-amber-400", badge: "text-amber-300 bg-amber-500/10 border-amber-500/40", accent: "bg-amber-500", row: "", rank: "P3" },
  LOW: { text: "text-emerald-400", badge: "text-emerald-300 bg-emerald-500/10 border-emerald-500/40", accent: "bg-emerald-500", row: "", rank: "P4" },
};
export const FALLBACK_SEVERITY = { text: "text-slate-400", badge: "text-slate-300 bg-slate-500/10 border-slate-500/40", accent: "bg-slate-500", row: "", rank: "P-" };

export const STATUS_STYLES = {
  ACTIVE: { badge: "text-blue-300 bg-blue-500/10 border-blue-500/40", dot: "bg-blue-400" },
  PENDING: { badge: "text-amber-300 bg-amber-500/10 border-amber-500/40", dot: "bg-amber-400" },
  RESOLVED: { badge: "text-emerald-300 bg-emerald-500/10 border-emerald-500/40", dot: "bg-emerald-400" },
  CLOSED: { badge: "text-slate-300 bg-slate-500/10 border-slate-500/40", dot: "bg-slate-400" },
};
export const FALLBACK_STATUS = STATUS_STYLES.CLOSED;

export const SEVERITY_FILTERS = [
  { value: "all", label: "All" },
  { value: "critical", label: "Critical" },
  { value: "high", label: "High" },
  { value: "medium", label: "Medium" },
  { value: "low", label: "Low" },
];

export const STATUS_FILTERS = [
  { value: "all", label: "All" },
  { value: "active", label: "Active" },
  { value: "pending", label: "Pending" },
  { value: "resolved", label: "Resolved" },
  { value: "closed", label: "Closed" },
];

// First matching keyword wins; anything unmatched is "other".
export const TYPE_FILTERS = [
  { value: "robbery", label: "Robbery", keywords: ["robbery"] },
  { value: "theft", label: "Theft", keywords: ["theft", "burglary"] },
  { value: "suspicious", label: "Suspicious Activity", keywords: ["suspicious"] },
  { value: "traffic", label: "Traffic Collision", keywords: ["traffic", "collision", "accident"] },
  { value: "fire", label: "Fire", keywords: ["fire"] },
  { value: "building", label: "Building Emergency", keywords: ["building", "collapse"] },
  { value: "missing", label: "Missing Person", keywords: ["missing"] },
  { value: "assault", label: "Assault", keywords: ["assault"] },
];

export const DEPARTMENT_FILTERS = [
  { value: "all", label: "All" },
  { value: "police", label: "Police" },
  { value: "fire", label: "Fire" },
  { value: "medical", label: "Medical" },
  { value: "traffic", label: "Traffic" },
];

export const DATE_FILTERS = [
  { value: "all", label: "All Time" },
  { value: "today", label: "Today" },
  { value: "24h", label: "Last 24 Hours" },
  { value: "7d", label: "Last 7 Days" },
  { value: "custom", label: "Custom" },
];

export const DEFAULT_FILTERS = {
  search: "", severity: "all", status: "all", type: "all", department: "all", date: "all", from: "", to: "",
};

export const STATUS_OPTIONS = ["ACTIVE", "PENDING", "RESOLVED", "CLOSED"];

export const TIMELINE_EVENT_OPTIONS = [
  { value: "INCIDENT_UPDATE", label: "Incident Update" },
  { value: "UNIT_DISPATCHED", label: "Unit Dispatched" },
  { value: "UNIT_ARRIVED", label: "Unit Arrived" },
  { value: "COORDINATION_REQUEST", label: "Coordination Request" },
];

export const CORE_DEPARTMENTS = ["Police", "Fire", "Medical", "Traffic"];
