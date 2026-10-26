import L from "leaflet";

const SVG = {
  incident: `<svg width="16" height="16" fill="#f87171" viewBox="0 0 24 24"><path d="M12 2L2 20h20L12 2zm0 4.5L18.5 18h-13L12 6.5zM11 10v4h2v-4h-2zm0 5v2h2v-2h-2z"/></svg>`,
  station: `<svg width="13" height="13" fill="#60a5fa" viewBox="0 0 24 24"><path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm0 10.99h7c-.53 4.12-3.28 7.79-7 8.94V12H5V6.3l7-3.11v8.8z"/></svg>`,
  crime: `<svg width="11" height="11" fill="#c084fc" viewBox="0 0 24 24"><path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm-1 5h2v4h-2zm0 6h2v2h-2z"/></svg>`,
};
const CONFIG = {
  incident: { color: "#ef4444", dot: 34 },
  station: { color: "#3b82f6", dot: 24 },
  crime: { color: "#a855f7", dot: 20 },
};

const cache = new Map();

export const getIcon = (kind, pulse = false) => {
  const key = `${kind}-${pulse}`;
  if (cache.has(key)) return cache.get(key);
  const { color, dot } = CONFIG[kind];
  const ring = dot + 14;
  const icon = L.divIcon({
    className: "custom-police-map-marker",
    html: `<div style="position:relative;width:${ring}px;height:${ring}px;display:flex;align-items:center;justify-content:center;">
      ${pulse ? `<div style="position:absolute;width:${ring}px;height:${ring}px;background:${color};opacity:.3;border-radius:50%;animation:incident-map-ping 1.6s cubic-bezier(0,0,.2,1) infinite;"></div>` : ""}
      <div style="width:${dot}px;height:${dot}px;background:#0f172a;border:2px solid ${color};border-radius:50%;display:flex;align-items:center;justify-content:center;box-shadow:0 0 ${kind === "incident" ? 14 : 5}px ${color};z-index:100;">${SVG[kind]}</div>
    </div>`,
    iconSize: [ring, ring],
    iconAnchor: [ring / 2, ring / 2],
    popupAnchor: [0, -(ring / 2)],
  });
  cache.set(key, icon);
  return icon;
};
