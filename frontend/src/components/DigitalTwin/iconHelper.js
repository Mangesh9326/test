import L from "leaflet";

/* ==========================================
   SVG ICONS
========================================== */
const SVG_ICONS = {
  hospital: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.4" stroke-linecap="round"><rect x="3" y="3" width="18" height="18" rx="3"/><path d="M12 7v10"/><path d="M7 12h10"/></svg>`,
  police_station: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2"><path d="M12 2L20 5V11C20 16 16.5 20 12 22C7.5 20 4 16 4 11V5L12 2Z"/></svg>`,
  cctv: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M4 9L17 4L20 10L7 15Z"/><path d="M16 6L22 3"/><path d="M10 14L12 20"/></svg>`,
  traffic_signal: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"><rect x="7" y="2" width="10" height="20" rx="2" fill="white"/><circle cx="12" cy="6" r="1.8" fill="#ef4444"/><circle cx="12" cy="12" r="1.8" fill="#f59e0b"/><circle cx="12" cy="18" r="1.8" fill="#22c55e"/></svg>`,
  ambulance: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M2 10H15V17H2Z"/><path d="M15 12H19L22 15V17H15"/><circle cx="6" cy="19" r="2"/><circle cx="18" cy="19" r="2"/><path d="M8 7V13"/><path d="M5 10H11"/></svg>`,
  police_vehicle: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M5 15L7 9H17L19 15"/><rect x="3" y="11" width="18" height="6" rx="2"/><circle cx="7" cy="18" r="2"/><circle cx="17" cy="18" r="2"/><rect x="10" y="6" width="4" height="2" fill="white"/></svg>`,
  fire_station: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M12 2C13 6 18 8 18 13A6 6 0 1 1 6 13C6 9 9 7 12 2Z"/></svg>`,
  weather_station: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M7 18A4 4 0 1 1 8 10A6 6 0 0 1 20 12A4 4 0 0 1 17 18Z"/></svg>`,
  water_sensor: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M12 2C12 2 5 10 5 14A7 7 0 0 0 19 14C19 10 12 2 12 2Z"/></svg>`,
  incident: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.6"><path d="M12 2L22 12L12 22L2 12Z"/><path d="M12 7V13"/><circle cx="12" cy="17" r="1"/></svg>`,
  // Fallback icon for unknown database entities
  default: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg>`
};

/* ==========================================
   CONFIGURATIONS (with default fallbacks)
========================================== */
const MARKER_COLORS = {
  hospital: "#ef4444",
  police_station: "#2563eb",
  cctv: "#10b981",
  traffic_signal: "#f59e0b",
  ambulance: "#ec4899",
  police_vehicle: "#6366f1",
  fire_station: "#f97316",
  weather_station: "#0ea5e9",
  water_sensor: "#06b6d4",
  incident: "#dc2626",
  default: "#64748b"
};

const SHAPES = {
  hospital: "16px",
  police_station: "8px",
  cctv: "30%",
  traffic_signal: "12px",
  ambulance: "50%",
  police_vehicle: "50%",
  fire_station: "10px",
  weather_station: "30%",
  water_sensor: "50% 50% 50% 0",
  incident: "4px",
  default: "50%"
};

const STATUS_COLORS = {
  online: "#22c55e",
  warning: "#f59e0b",
  critical: "#ef4444",
  offline: "#6b7280"
};

/* ==========================================
   ICON GENERATOR
========================================== */
export const getEnterpriseIcon = (type, status = "online") => {
  // Fail-safe fallbacks prevent the map from crashing on unknown entity types
  const color = MARKER_COLORS[type] || MARKER_COLORS.default;
  const svg = SVG_ICONS[type] || SVG_ICONS.default;
  const radius = SHAPES[type] || SHAPES.default;
  const statusColor = STATUS_COLORS[status] || STATUS_COLORS.offline;

  const isIncident = type === "incident";
  
  // Conditionally render the ripple to keep DOM lightweight for static markers
  const rippleHTML = isIncident 
    ? `<span class="ripple-effect" style="background:${color};"></span>` 
    : '';

  return L.divIcon({
    className: "custom-leaflet-icon", // Replaced empty string to prevent Leaflet default class quirks
    html: `
      <div class="enterprise-marker ${isIncident ? "pulse-marker" : ""}" 
           style="width: 40px; height: 40px; position: relative; display: flex; align-items: center; justify-content: center;">
          
          ${rippleHTML}

          <div style="
              position: relative;
              z-index: 10;
              width: 32px; 
              height: 32px; 
              border-radius: ${radius}; 
              background: ${color}; 
              display: flex; 
              justify-content: center; 
              align-items: center; 
              border: 2px solid rgba(255,255,255,0.95);
              box-shadow: 0 0 0 3px rgba(255,255,255,0.15), 0 0 12px ${color}, 0 6px 18px rgba(0,0,0,0.45);
          ">
              ${svg}
          </div>

          <span style="
              position: absolute; 
              top: 2px; 
              right: 2px; 
              z-index: 20;
              width: 10px; 
              height: 10px; 
              border-radius: 50%; 
              background: ${statusColor}; 
              border: 2px solid #ffffff;
              box-shadow: 0 0 4px rgba(0,0,0,0.5);
          "></span>
      </div>
    `,
    iconSize: [40, 40],
    iconAnchor: [20, 20],
    popupAnchor: [0, -20],
  });
};