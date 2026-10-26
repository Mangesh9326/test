import React, { useEffect } from "react";
import L from "leaflet";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";

// Custom Camera Marker Generator
const createTrafficCameraIcon = (status, isSelected) => {
  const isOnline = status === "Online" || status === "Live";
  const ringSize = isSelected ? 48 : 36;
  const dotSize = isSelected ? 34 : 26;
  const iconColor = isOnline ? "#10b981" : "#ef4444";
  const glowColor = isOnline ? "rgba(16, 185, 129, 0.45)" : "rgba(239, 68, 68, 0.45)";

  return L.divIcon({
    className: "custom-traffic-camera-marker",
    html: `
      <div style="position: relative; width: ${ringSize}px; height: ${ringSize}px; display: flex; align-items: center; justify-content: center;">
        <div style="position: absolute; width: ${ringSize}px; height: ${ringSize}px; background: ${glowColor}; border-radius: 50%; ${isSelected ? 'animation: ping 1.2s cubic-bezier(0, 0, 0.2, 1) infinite;' : ''}"></div>
        <div style="width: ${dotSize}px; height: ${dotSize}px; background: #0f172a; border: 2px solid ${iconColor}; border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 ${isSelected ? '16px' : '8px'} ${iconColor}; z-index: ${isSelected ? 100 : 10};">
          <svg style="width: 14px; height: 14px; fill: ${iconColor};" viewBox="0 0 24 24">
            <path d="M15 8v8H5V8h10m1-2H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4V7c0-.55-.45-1-1-1z"/>
          </svg>
        </div>
      </div>
    `,
    iconSize: [ringSize, ringSize],
    iconAnchor: [ringSize / 2, ringSize / 2],
    popupAnchor: [0, -ringSize / 2],
  });
};

// Camera Map View Controller
const CameraMapViewController = ({ center, zoom }) => {
  const map = useMap();
  const lat = center ? center[0] : null;
  const lng = center ? center[1] : null;

  useEffect(() => {
    if (lat !== null && lng !== null) {
      map.flyTo([lat, lng], zoom || 16, { duration: 1.2 });
    }
  }, [lat, lng, zoom, map]);

  return null;
};

export default function TrafficCameraMap({
  cameras = [],
  selectedCamera = null,
  onCameraSelect = () => {},
}) {
  const defaultCenter = [19.076, 72.8777]; // Mumbai center
  const focusedCenter = selectedCamera
    ? [selectedCamera.latitude, selectedCamera.longitude]
    : defaultCenter;
  const focusedZoom = selectedCamera ? 16 : 12;

  return (
    <div className="relative w-full h-full bg-slate-950 overflow-hidden">
      <MapContainer
        center={defaultCenter}
        zoom={12}
        className="absolute inset-0 z-0"
        zoomControl={false}
      >
        <CameraMapViewController center={focusedCenter} zoom={focusedZoom} />

        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Removed MarkerClusterGroup so all cameras render individually */}
        {cameras.map((cam) => {
          const isSelected = selectedCamera?.camera_id === cam.camera_id;
          return (
            <Marker
              key={cam.camera_id}
              position={[cam.latitude, cam.longitude]}
              icon={createTrafficCameraIcon(cam.status, isSelected)}
              zIndexOffset={isSelected ? 1000 : 0}
              eventHandlers={{
                click: () => onCameraSelect(cam),
              }}
            >
              <Popup className="enterprise-popup border-t-4 border-blue-500">
                <div className="w-64 bg-slate-900 text-slate-100 p-3 rounded-lg shadow-xl font-sans">
                  <div className="flex justify-between items-center border-b border-slate-700 pb-1 mb-2">
                    <span className="font-bold text-sm text-blue-400 font-mono">
                      {cam.camera_code}
                    </span>
                    <span
                      className={`text-[9px] px-1.5 py-0.5 rounded font-bold uppercase ${
                        cam.status === "Online"
                          ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                          : "bg-red-500/20 text-red-400 border border-red-500/30"
                      }`}
                    >
                      {cam.status}
                    </span>
                  </div>
                  <div className="text-xs font-semibold text-slate-200 mb-1">
                    {cam.name}
                  </div>
                  <div className="text-[11px] text-slate-400 font-mono space-y-0.5">
                    <div>Road: {cam.road_name}</div>
                    <div>Axis: {cam.direction}</div>
                    <div>
                      Coords: {cam.latitude.toFixed(4)}°, {cam.longitude.toFixed(4)}°
                    </div>
                  </div>
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>
    </div>
  );
}