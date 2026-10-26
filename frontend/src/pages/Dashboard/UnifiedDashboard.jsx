import React, { useState, useEffect, useMemo } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import DigitalTwinMap from '../../components/DigitalTwin/MapWidget';
import { useMapStore } from '../../store/useMapStore';

// =========================================================================
// ICONS (inline, dependency-free)
// =========================================================================

const IconCar = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M3 17V9a1 1 0 011-1h1.5l2-4h9l2 4H21a1 1 0 011 1v8H3z" />
    <circle cx="7.5" cy="17" r="1.7" fill="currentColor" stroke="none" />
    <circle cx="16.5" cy="17" r="1.7" fill="currentColor" stroke="none" />
  </svg>
);
const IconPlus = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
  </svg>
);
const IconShield = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
  </svg>
);
const IconFlame = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M17.657 18.657A8 8 0 116.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.657 7.343a7.975 7.975 0 010 11.314z" />
  </svg>
);
const IconBolt = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
  </svg>
);
const IconMegaphone = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M11 5.882V19.24a1.76 1.76 0 01-3.417.592L5.436 13.68M18 13a3 3 0 100-6M5.436 13.683A4.001 4.001 0 017 6h1.832c4.1 0 7.625-1.234 9.168-3v14c-1.543-1.766-5.067-3-9.168-3H7a4 4 0 01-1.564-.317z" />
  </svg>
);
const IconPin = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M17.657 16.657L13.414 20.9a2 2 0 01-2.828 0l-4.243-4.243a8 8 0 1111.314 0z" />
    <circle cx="12" cy="11" r="2.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);
const IconUsers = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M17 20h5v-2a4 4 0 00-3-3.87M9 20H4v-2a4 4 0 013-3.87m5-2.13a4 4 0 100-8 4 4 0 000 8zm7 1a4 4 0 10-2.516-7.14" />
  </svg>
);
const IconVideo = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l5-3v10l-5-3M4 6h9a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V8a2 2 0 012-2z" />
  </svg>
);
const IconChevron = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
  </svg>
);
const IconTarget = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="8" /><circle cx="12" cy="12" r="4" /><circle cx="12" cy="12" r="0.8" fill="currentColor" />
  </svg>
);

// =========================================================================
// STATIC STYLE MAPS
// =========================================================================

const COLOR_STYLES = {
  indigo: { border: 'border-t-indigo-500/70', text: 'text-indigo-400', bg: 'bg-indigo-500/10', ring: 'border-indigo-500/30', bar: 'bg-indigo-500', glow: 'shadow-[0_0_15px_rgba(99,102,241,0.15)]' },
  red: { border: 'border-t-red-500/70', text: 'text-red-400', bg: 'bg-red-500/10', ring: 'border-red-500/30', bar: 'bg-red-500', glow: 'shadow-[0_0_15px_rgba(239,68,68,0.15)]' },
  emerald: { border: 'border-t-emerald-500/70', text: 'text-emerald-400', bg: 'bg-emerald-500/10', ring: 'border-emerald-500/30', bar: 'bg-emerald-500', glow: 'shadow-[0_0_15px_rgba(16,185,129,0.15)]' },
  blue: { border: 'border-t-blue-500/70', text: 'text-blue-400', bg: 'bg-blue-500/10', ring: 'border-blue-500/30', bar: 'bg-blue-500', glow: 'shadow-[0_0_15px_rgba(59,130,246,0.15)]' },
  orange: { border: 'border-t-orange-500/70', text: 'text-orange-400', bg: 'bg-orange-500/10', ring: 'border-orange-500/30', bar: 'bg-orange-500', glow: 'shadow-[0_0_15px_rgba(249,115,22,0.15)]' },
  purple: { border: 'border-t-purple-500/70', text: 'text-purple-400', bg: 'bg-purple-500/10', ring: 'border-purple-500/30', bar: 'bg-purple-500', glow: 'shadow-[0_0_15px_rgba(168,85,247,0.15)]' },
  teal: { border: 'border-t-teal-500/70', text: 'text-teal-400', bg: 'bg-teal-500/10', ring: 'border-teal-500/30', bar: 'bg-teal-500', glow: 'shadow-[0_0_15px_rgba(20,184,166,0.15)]' },
};

const SEVERITY_STYLES = {
  Critical: { text: 'text-red-400', bg: 'bg-red-500/10', border: 'border-red-500/30', dot: 'bg-red-500' },
  High: { text: 'text-orange-400', bg: 'bg-orange-500/10', border: 'border-orange-500/30', dot: 'bg-orange-500' },
  Medium: { text: 'text-yellow-400', bg: 'bg-yellow-500/10', border: 'border-yellow-500/30', dot: 'bg-yellow-500' },
  Low: { text: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', dot: 'bg-emerald-500' },
};

const DEPARTMENTS = [
  { id: 'traffic', name: 'Traffic Dept', status: 'Alert', color: 'red', icon: IconCar, load: 78 },
  { id: 'hospital', name: 'Hospital Dept', status: 'Deploying', color: 'emerald', icon: IconPlus, load: 45 },
  { id: 'police', name: 'Police Dept', status: 'Coordinating', color: 'blue', icon: IconShield, load: 60 },
  { id: 'fire', name: 'Fire Dept', status: 'Standby', color: 'orange', icon: IconFlame, load: 12 },
  { id: 'utility', name: 'Utility Dept', status: 'Nominal', color: 'purple', icon: IconBolt, load: 30 },
  { id: 'citizen', name: 'Citizen Services', status: 'Alerting', color: 'teal', icon: IconMegaphone, load: 88 },
];

// =========================================================================
// DATA NORMALIZATION
// =========================================================================

const normalizeIncident = (inc) => {
  let payload = {};
  
  try {
    const dataString = inc.timeline_events || inc.description;
    if (dataString) {
      payload = typeof dataString === 'string' ? JSON.parse(dataString) : dataString;
    }
  } catch (e) {
    payload = { description: inc.timeline_events || inc.description }; 
  }

  const lat = parseFloat(inc.latitude);
  const lng = parseFloat(inc.longitude);

  let videoUrl = null;
  if (payload.videoUrl) {
    videoUrl = payload.videoUrl;
  } else if (payload.video_file) {
    videoUrl = `http://localhost:5000/videos/${payload.video_file}`;
  } else if (inc.scenario_id) {
    videoUrl = `http://localhost:5000/videos/${inc.scenario_id}.mp4`;
  }

  let conf = 'N/A';
  if (payload.confidence) {
    conf = payload.confidence <= 1 ? Math.round(payload.confidence * 100) : payload.confidence;
  }

  const severity = payload.severity || inc.severity || 'High';
  const type = inc.incident_type || 'System Event';
  const typeLower = type.toLowerCase();
  const descLower = (payload.description || '').toLowerCase();

  const agents = [{ name: 'Decision Coordinator', color: 'indigo' }];
  
  if (typeLower.includes('traffic') || typeLower.includes('collision') || descLower.includes('vehicle') || descLower.includes('car')) {
    agents.push({ name: 'Traffic Agent', color: 'red' });
  }
  
  if (typeLower.includes('robbery') || typeLower.includes('police') || typeLower.includes('pursuit') || typeLower.includes('violence') || severity === 'Critical') {
    agents.push({ name: 'Police Agent', color: 'blue' });
  }
  
  if (severity === 'Critical' || severity === 'High' || typeLower.includes('fire') || typeLower.includes('accident')) {
    agents.push({ name: 'Hospital Agent', color: 'emerald' });
    if (typeLower.includes('fire') || typeLower.includes('explosion') || descLower.includes('smoke')) {
        agents.push({ name: 'Fire Agent', color: 'orange' });
    }
  }
  
  if (typeLower.includes('weather') || typeLower.includes('public') || severity === 'Critical') {
    agents.push({ name: 'Citizen Agent', color: 'teal' });
  }
  
  if (typeLower.includes('utility') || typeLower.includes('power')) {
    agents.push({ name: 'Utility Agent', color: 'purple' });
  }

  const actionPlan = [];
  const units = payload.unitsAssigned || (severity === 'Critical' ? 5 : 2);
  
  actionPlan.push(`Dispatch ${units !== 'N/A' ? units : 'Available'} Responders (Unified Multi-Agent Sync)`);
  
  if (agents.some(a => a.name === 'Police Agent')) {
    const coordsStr = !isNaN(lat) && !isNaN(lng) ? `${lat.toFixed(4)}° N, ${lng.toFixed(4)}° E` : 'target perimeter';
    actionPlan.push(`Isolate Incident Perimeter at ${coordsStr} (Police Agent)`);
  }
  
  if (agents.some(a => a.name === 'Traffic Agent')) {
    actionPlan.push(`Initiate Dynamic Traffic Diversion Protocol (Traffic Agent)`);
    if (severity === 'Critical') {
        actionPlan.push(`Activate Emergency Green Corridor (Traffic Agent)`);
    }
  }
  
  if (agents.some(a => a.name === 'Fire Agent')) {
    actionPlan.push(`Deploy Specialized Fire Suppression Units (Fire Agent)`);
  }
  
  if (agents.some(a => a.name === 'Citizen Agent')) {
    actionPlan.push(`Broadcast Area-Wide Mobile Safety Alerts (Citizen Agent)`);
  }

  return {
    id: String(inc.incident_id),
    title: type.replace(/_/g, ' '),
    type: type,
    severity: severity,
    status: inc.status || 'Active',
    reportedAt: inc.created_at ? new Date(inc.created_at).toLocaleTimeString() : 'Live Feed',
    address: payload.location || payload.address || 'Camera / Sensor Node',
    coords: !isNaN(lat) && !isNaN(lng) ? `${lat.toFixed(4)}° N, ${lng.toFixed(4)}° E` : 'N/A',
    lat: !isNaN(lat) ? lat : null,
    lng: !isNaN(lng) ? lng : null,
    distance: payload.distance || 'N/A',
    unitsAssigned: units,
    camera: payload.camera || 'Live CCTV Edge Node',
    aiTag: payload.description || payload.aiTag || 'AI Vision Trigger',
    confidence: conf,
    congestion: payload.congestion || 'N/A',
    videoUrl: videoUrl,
    source: inc.upload_id ? 'Live' : 'Scenario',
    activeAgents: agents,
    actionPlan: actionPlan
  };
};

// =========================================================================
// DEPARTMENT PANEL CONTENT
// =========================================================================

const TrafficPanel = () => (
  <div className="p-3 text-sm flex flex-col gap-2">
    <div className="flex justify-between text-gray-300">
      <span className="text-gray-500">Expected Congestion:</span>
      <span className="font-mono text-yellow-400">42%</span>
    </div>
    <div className="flex justify-between text-gray-300">
      <span className="text-gray-500">Diversion Routes:</span>
      <span className="text-xs">Ring Rd, Station Rd</span>
    </div>
  </div>
);

const HospitalPanel = () => (
  <div className="p-3 text-sm flex flex-col gap-2">
    <div className="flex justify-between text-gray-300">
      <span className="text-gray-500">Ambulances Dispatched:</span>
      <span className="font-mono text-blue-400">2</span>
    </div>
    <div className="flex justify-between text-gray-300">
      <span className="text-gray-500">Available ICU Beds:</span>
      <span className="font-mono text-emerald-400">5</span>
    </div>
  </div>
);

const PolicePanel = () => (
  <div className="p-3 text-sm flex flex-col gap-2">
    <div className="flex justify-between text-gray-300">
      <span className="text-gray-500">Officers Deployed:</span>
      <span className="font-mono">4</span>
    </div>
    <div className="flex justify-between text-gray-300">
      <span className="text-gray-500">Intersection Status:</span>
      <span className="text-red-400 text-xs uppercase tracking-wider">Closed</span>
    </div>
  </div>
);

const FirePanel = () => (
  <div className="p-3 text-sm flex flex-col gap-2">
    <div className="flex justify-between text-gray-300">
      <span className="text-gray-500">Fire/Fuel Risk:</span>
      <span className="font-mono text-emerald-400">LOW</span>
    </div>
    <div className="flex justify-between text-gray-300">
      <span className="text-gray-500">Trucks Deployed:</span>
      <span className="font-mono">0</span>
    </div>
  </div>
);

const UtilityPanel = () => (
  <div className="p-3 text-sm flex flex-col gap-2">
    <div className="flex justify-between text-gray-300">
      <span className="text-gray-500">Power Stations:</span>
      <span className="font-mono text-emerald-400">Stable</span>
    </div>
    <div className="flex justify-between text-gray-300">
      <span className="text-gray-500">Streetlights/Water:</span>
      <span className="font-mono">Nominal</span>
    </div>
  </div>
);

const CitizenPanel = () => (
  <div className="p-3 text-sm flex flex-col gap-2">
    <div className="flex justify-between text-gray-300">
      <span className="text-gray-500">Environment:</span>
      <span className="text-xs">Normal</span>
    </div>
    <div className="flex justify-between text-gray-300">
      <span className="text-gray-500">Broadcast Issued:</span>
      <span className="text-xs text-red-400">"Emergency Active"</span>
    </div>
  </div>
);

const DEPT_PANEL_MAP = {
  traffic: TrafficPanel,
  hospital: HospitalPanel,
  police: PolicePanel,
  fire: FirePanel,
  utility: UtilityPanel,
  citizen: CitizenPanel,
};

// =========================================================================
// UI COMPONENTS
// =========================================================================

const DepartmentCard = ({ dept }) => {
  const s = COLOR_STYLES[dept.color];
  const PanelBody = DEPT_PANEL_MAP[dept.id];
  const DeptIcon = dept.icon;
  return (
    <div className={`glass-panel flex flex-col border-t-2 ${s.border} hover:bg-city-800/80 hover:-translate-y-0.5 transition-all duration-200 shadow-md`}>
      <div className="border-b border-city-700/50 p-2.5 bg-city-800/30 rounded-t-xl flex justify-between items-center">
        <span className="flex items-center gap-1.5 text-[10px] font-bold text-gray-400 uppercase tracking-widest">
          <DeptIcon className={`w-3.5 h-3.5 ${s.text}`} />
          {dept.name}
        </span>
        <span className={`${s.text} text-[10px] font-bold`}>{dept.status}</span>
      </div>
      <PanelBody />
      <div className="px-3 pb-3 mt-auto">
        <div className="flex justify-between text-[9px] text-gray-500 mb-1 font-mono">
          <span>RESOURCE LOAD</span><span>{dept.load}%</span>
        </div>
        <div className="h-1.5 w-full bg-city-800 rounded-full overflow-hidden">
          <div className={`h-full ${s.bar} rounded-full transition-all duration-700`} style={{ width: `${dept.load}%` }}></div>
        </div>
      </div>
    </div>
  );
};

const IncidentListCard = ({ incident, isSelected, onSelect }) => {
  const s = SEVERITY_STYLES[incident.severity] || SEVERITY_STYLES['Medium'];
  return (
    <button
      onClick={() => onSelect(incident.id)}
      className={`w-full text-left p-3 rounded-lg border transition-all duration-150 flex flex-col gap-1.5
        ${isSelected ? `${s.bg} ${s.border} shadow-[0_0_12px_rgba(0,0,0,0.25)] scale-[1.02]` : 'bg-black/20 border-city-700/40 hover:border-city-600 hover:bg-city-800/50'}`}
    >
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1.5">
          <span className={`w-1.5 h-1.5 rounded-full ${s.dot} ${incident.severity === 'Critical' ? 'animate-pulse' : ''}`}></span>
          <span className={`text-[9px] font-bold uppercase tracking-widest ${s.text}`}>{incident.severity}</span>
        </span>
        <span className="text-[9px] text-gray-500 font-mono">{incident.reportedAt}</span>
      </div>
      <div className="text-sm font-semibold text-gray-200 flex items-center justify-between gap-1">
        <span className="truncate">{incident.title}</span>
        <IconChevron className={`w-3.5 h-3.5 shrink-0 transition-transform ${isSelected ? 'text-gray-300 translate-x-0.5' : 'text-gray-600'}`} />
      </div>
      <div className="text-[11px] text-gray-500 flex items-center gap-1">
        <IconPin className="w-3 h-3 shrink-0" />
        <span className="truncate">{incident.address}</span>
      </div>
    </button>
  );
};

const IncidentsSidebar = ({ incidents, selectedId, onSelect, tall = false }) => {
  const criticalCount = incidents.filter((i) => i.severity === 'Critical' || i.severity === 'High').length;
  return (
    <div className={`glass-panel flex flex-col border-city-700/80 shadow-lg overflow-hidden ${tall ? 'h-full' : 'h-[360px]'}`}>
      <div className="border-b border-city-700/50 p-3.5 bg-city-800/30 rounded-t-xl flex flex-col gap-1.5 shrink-0">
        <span className="text-xs font-bold text-gray-300 uppercase tracking-widest flex items-center gap-2">
          <IconTarget className="w-4 h-4 text-red-400" />
          Active Incidents
        </span>
        <span className="text-[9px] bg-red-500/10 text-red-400 px-2 py-1 rounded border border-red-500/30 font-mono w-fit">
          {incidents.length} LIVE · {criticalCount} CRIT
        </span>
      </div>
      <div className="flex-grow overflow-y-auto p-2.5 flex flex-col gap-2 custom-scrollbar">
        {incidents.map((incident) => (
          <IncidentListCard
            key={incident.id}
            incident={incident}
            isSelected={String(incident.id) === String(selectedId)}
            onSelect={onSelect}
          />
        ))}
      </div>
    </div>
  );
};

const DetailRow = ({ label, value, valueClass = 'text-gray-200' }) => (
  <div className="flex items-center justify-between py-1.5 border-b border-city-700/30 last:border-0">
    <span className="text-[11px] text-gray-500 uppercase tracking-wide">{label}</span>
    <span className={`text-xs font-mono ${valueClass}`}>{value}</span>
  </div>
);

const IncidentDetailsPanel = ({ incident }) => {
  const s = SEVERITY_STYLES[incident.severity] || SEVERITY_STYLES['Medium'];

  const handleResolve = async () => {
    if (!window.confirm("Are you sure you want to resolve and permanently delete this incident from the system?")) return;

    try {
        // Updated to use the DELETE method and correct endpoint
        const response = await fetch(`http://localhost:5000/api/simulation/incidents/${incident.id}`, { 
            method: 'DELETE' 
        });
        
        if (response.ok) {
            alert("Incident resolved and removed from database. Refreshing dashboard.");
            window.location.href = '/'; 
        } else {
            console.error("Failed to delete incident in DB");
        }
    } catch(e) {
        console.error("Failed to resolve", e);
    }
  };

  return (
    <div className="glass-panel border-city-700/80 shadow-lg overflow-hidden shrink-0">
      <div className="border-b border-city-700/50 p-3.5 bg-city-800/30 flex items-center justify-between">
        <span className="text-xs font-bold text-gray-300 uppercase tracking-widest">Incident Details &amp; Location</span>
        <div className="flex items-center gap-2">
            <button 
              onClick={handleResolve} 
              className="text-[9px] font-bold bg-city-700 hover:bg-red-500 hover:text-white text-gray-300 px-2 py-1 rounded border border-city-600 transition-colors cursor-pointer"
            >
              MARK RESOLVED
            </button>
            <span className={`text-[9px] px-2 py-1 rounded border font-bold uppercase tracking-widest ${s.bg} ${s.border} ${s.text}`}>
              {incident.severity}
            </span>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-0 divide-x divide-city-700/40">
        <div className="p-4 flex flex-col gap-1">
          <div className="text-lg font-bold text-gray-100 mb-1">{incident.title}</div>
          <div className="text-[11px] text-gray-500 font-mono mb-2">{incident.id} · {incident.type}</div>
          <DetailRow label="Status" value={incident.status} valueClass={s.text} />
          <DetailRow label="Reported" value={incident.reportedAt} />
          <DetailRow label="Units Assigned" value={incident.unitsAssigned} valueClass="text-blue-400" />
          <DetailRow 
            label="Congestion Impact" 
            value={incident.congestion !== 'N/A' ? `${incident.congestion}%` : 'N/A'} 
            valueClass={incident.congestion !== 'N/A' ? "text-yellow-400" : "text-gray-500"} 
          />
        </div>
        <div className="p-4 flex flex-col gap-2 relative overflow-hidden">
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:22px_22px] opacity-20 pointer-events-none"></div>
          <div className="relative z-10 flex items-center gap-2 text-gray-300 text-sm font-semibold">
            <IconPin className="w-4 h-4 text-red-400" />
            Location
          </div>
          <div className="relative z-10 flex flex-col gap-1.5 mt-1">
            <div className="text-xs text-gray-300">{incident.address}</div>
            <div className="text-[11px] text-gray-500 font-mono">{incident.coords}</div>
            <div className="flex items-center gap-1.5 text-[11px] text-emerald-400 mt-1">
              <IconUsers className="w-3.5 h-3.5" />
              {incident.distance}
            </div>
          </div>
          <div className="relative z-10 mt-auto flex justify-center pt-3">
            <div className="relative w-10 h-10 flex items-center justify-center">
              <span className="absolute w-10 h-10 rounded-full bg-red-500/20 animate-ping"></span>
              <span className="relative w-3 h-3 rounded-full bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.9)]"></span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const FootagePanel = ({ incident }) => {
  const [now, setNow] = useState(new Date());
  
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);
  
  const timestamp = now.toLocaleTimeString('en-IN', { hour12: false });

  return (
    <div className="glass-panel border-city-700/80 shadow-lg overflow-hidden shrink-0">
      <div className="border-b border-city-700/50 p-3.5 bg-city-800/30 flex items-center justify-between">
        <span className="text-xs font-bold text-gray-300 uppercase tracking-widest flex items-center gap-2">
          <IconVideo className="w-4 h-4 text-blue-400" />
          Incident Footage
        </span>
        <span className="text-[9px] bg-blue-500/10 text-blue-300 px-2 py-1 rounded border border-blue-500/30 font-mono">
          AI VISION
        </span>
      </div>

      <div className="relative h-64 bg-black overflow-hidden flex items-center justify-center">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:32px_32px] opacity-20 z-0 pointer-events-none"></div>
        
        {incident.videoUrl ? (
          <video 
            key={incident.id} 
            src={incident.videoUrl} 
            controls 
            autoPlay 
            muted 
            loop 
            className="absolute inset-0 w-full h-full object-contain z-10" 
          />
        ) : (
          <div className="z-10 flex flex-col items-center gap-2 text-gray-600">
            <IconVideo className="w-8 h-8 opacity-50" />
            <span className="text-xs font-mono uppercase tracking-widest">Video Feed N/A</span>
          </div>
        )}

        {/* Top Overlay HUD */}
        <div className="absolute top-0 inset-x-0 flex items-center justify-between p-3 z-20 pointer-events-none">
          <span className="flex items-center gap-1.5 text-[10px] font-mono text-gray-300 bg-black/60 px-2 py-1 rounded border border-city-700/60">
            <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse"></span>
            REC
            <span className="text-gray-500">|</span>
            {incident.camera}
          </span>
          <span className="text-[10px] font-mono text-gray-300 bg-black/60 px-2 py-1 rounded border border-city-700/60">
            {timestamp}
          </span>
        </div>

        {/* AI Tag Overlay */}
        {incident.videoUrl && incident.aiTag && incident.aiTag !== 'N/A' && (
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-20 pointer-events-none">
            <span className="text-[10px] font-mono bg-emerald-500/90 text-black px-1.5 py-0.5 rounded whitespace-nowrap shadow-lg">
              {incident.aiTag} {incident.confidence !== 'N/A' && `· ${incident.confidence}%`}
            </span>
          </div>
        )}

        {/* Note: The static fake controls (SVG play button + custom progress bar) 
            were removed from here so they don't overlap with the native browser controls. */}
      </div>
    </div>
  );
};

const ConfidenceGauge = ({ value }) => {
  const radius = 26;
  const circumference = 2 * Math.PI * radius;
  const numericValue = value === 'N/A' ? 0 : Number(value);
  const offset = circumference - (numericValue / 100) * circumference;
  
  return (
    <svg width="64" height="64" viewBox="0 0 64 64" className="shrink-0">
      <circle cx="32" cy="32" r={radius} stroke="#1e293b" strokeWidth="6" fill="none" />
      <circle
        cx="32" cy="32" r={radius}
        stroke="url(#confidenceGradient)" strokeWidth="6" fill="none"
        strokeDasharray={circumference} strokeDashoffset={offset}
        strokeLinecap="round" transform="rotate(-90 32 32)"
        className="transition-all duration-700"
      />
      <defs>
        <linearGradient id="confidenceGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#6366f1" />
          <stop offset="100%" stopColor="#22d3ee" />
        </linearGradient>
      </defs>
      <text x="32" y="36" textAnchor="middle" fontSize="13" fill="#e2e8f0" fontWeight="700">
        {value === 'N/A' ? 'N/A' : `${value}%`}
      </text>
    </svg>
  );
};

const DecisionCommander = ({ incident }) => (
  <div className="p-4 flex flex-col gap-3 h-full">
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></div>
        <span className="text-xs text-red-400 font-bold uppercase tracking-widest">Global Priority: {incident.severity}</span>
      </div>
      <ConfidenceGauge value={incident.confidence} />
    </div>

    <div className="flex flex-wrap gap-1.5">
      {incident.activeAgents && incident.activeAgents.map((a) => {
        const s = COLOR_STYLES[a.color] || COLOR_STYLES['blue'];
        return (
          <span key={a.name} className={`text-[9px] font-bold px-2 py-1 rounded-full border flex items-center gap-1 ${s.bg} ${s.ring} ${s.text}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${s.bar}`}></span>
            {a.name}
          </span>
        );
      })}
    </div>

    <div className="bg-black/40 border border-city-700/50 rounded-lg p-4 mt-2">
      <div className="text-sm font-bold text-gray-200 mb-1">Target: {incident.title}</div>
      <div className="text-xs text-gray-500 mb-4">Multi-Agent Unified Action Plan Executing:</div>
      
      <ul className="space-y-3 text-sm text-gray-300">
        {incident.actionPlan && incident.actionPlan.map((action, idx) => (
          <li key={idx} className="flex items-start gap-2">
            <span className="text-emerald-500 mt-0.5">✓</span>
            <span>{action}</span>
          </li>
        ))}
      </ul>
    </div>
  </div>
);

const LLMAssistant = ({ incident }) => (
  <div className="p-4 flex flex-col h-full gap-3">
    <div className="flex-grow overflow-y-auto space-y-3">
      <div className="flex justify-end">
        <div className="bg-city-700/50 text-gray-200 text-sm px-3 py-2 rounded-lg rounded-tr-none max-w-[85%]">
          What happened?
        </div>
      </div>
      <div className="flex justify-start">
        <div className="bg-blue-900/20 border border-blue-500/30 text-blue-100 text-sm px-3 py-2 rounded-lg rounded-tl-none max-w-[95%] leading-relaxed">
          <span className="font-bold text-blue-400 text-xs block mb-1">Grok AI System:</span>
          A <strong>{incident.type.toLowerCase()}</strong> was automatically detected at {incident.address} ({incident.coords}). Confidence level is {incident.confidence !== 'N/A' ? `${incident.confidence}%` : 'unknown'}. {incident.unitsAssigned !== 'N/A' ? `${incident.unitsAssigned} units have` : 'Units have'} been dispatched. The multi-agent unified response plan is currently active, coordinated among: {incident.activeAgents ? incident.activeAgents.map(a => a.name).join(', ') : 'City Responders'}.
        </div>
      </div>
    </div>
  </div>
);

// =========================================================================
// MAIN LAYOUT COMPONENT
// =========================================================================

export default function UnifiedDashboard() {
  const location = useLocation();
  const navigate = useNavigate();
  const [now, setNow] = useState(new Date());
  
  const { incidents: dbIncidents } = useMapStore();

  const displayIncidents = useMemo(() => {
    if (!dbIncidents || dbIncidents.length === 0) return [];

    const sortedDb = [...dbIncidents].sort((a, b) => b.incident_id - a.incident_id);

    return sortedDb.map(normalizeIncident);
  }, [dbIncidents]);

  // FIX: Filter out resolved/inactive incidents so they disappear from the active sidebar and map
  const activeIncidents = useMemo(() => {
    return displayIncidents.filter(inc => inc.status && inc.status.toLowerCase() !== 'resolved' && inc.status.toLowerCase() !== 'inactive');
  }, [displayIncidents]);

  const selectedId = new URLSearchParams(location.search).get('incidentId');

  // FIX: Auto-Default Navigation to the Newest Incident if no URL param is present
  useEffect(() => {
    if (!selectedId && activeIncidents.length > 0) {
      navigate(`/?incidentId=${activeIncidents[0].id}`, { replace: true });
    }
  }, [selectedId, activeIncidents, navigate]);

  const handleSelectIncident = (id) => {
    navigate(`/?incidentId=${id}`, { replace: true });
  };

  const selectedIncident = useMemo(
    () => activeIncidents.find((i) => i.id === selectedId) || null,
    [activeIncidents, selectedId]
  );
  
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const criticalCount = activeIncidents.filter((i) => i.severity === 'Critical' || i.severity === 'High').length;

  const focusedCenter = selectedIncident && selectedIncident.lat && selectedIncident.lng 
    ? [selectedIncident.lat, selectedIncident.lng] 
    : null;

  return (
    <div className="h-screen overflow-hidden bg-city-900 p-6 flex flex-col gap-6 text-gray-100 font-sans">
      
      <header className="glass-panel p-5 flex justify-between items-center shrink-0 shadow-lg">
        <div>
          <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-teal-300">
            Unified Command Center
          </h1>
          <p className="text-xs text-gray-400 uppercase tracking-widest mt-1.5">
            <span className="text-emerald-400 font-bold mr-2">● LIVE</span>
            {now.toLocaleTimeString('en-IN', { hour12: false })} | Scenario: {selectedIncident ? `${selectedIncident.source} Edge Processing` : 'Standby'}
          </p>
        </div>
        <div className="flex space-x-4 text-sm font-medium">
          <div className="bg-red-500/10 text-red-400 px-5 py-2.5 rounded-lg border border-red-500/30 shadow-[0_0_15px_rgba(239,68,68,0.15)] flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-ping"></span>
            {criticalCount} Critical Alert{criticalCount !== 1 ? 's' : ''}
          </div>
          <div className="bg-indigo-500/10 text-indigo-300 px-5 py-2.5 rounded-lg border border-indigo-500/30">
            AI Confidence: {selectedIncident && selectedIncident.confidence !== 'N/A' ? `${selectedIncident.confidence}%` : 'N/A'}
          </div>
          <div className="bg-blue-500/10 text-blue-400 px-5 py-2.5 rounded-lg border border-blue-500/30">
            Agents Active: {selectedIncident?.activeAgents ? selectedIncident.activeAgents.length : 0}
          </div>
        </div>
      </header>

      <main className="grid grid-cols-12 gap-6 flex-1 min-h-0">
        <section className="col-span-10 h-full min-h-0 overflow-y-auto pr-1 flex flex-col gap-6 custom-scrollbar">
          
          <div className="glass-panel h-80 relative overflow-hidden flex flex-col border-city-700/80 shadow-lg shrink-0">
            <div className="absolute top-3 left-3 z-20 px-2 py-1 bg-black/80 rounded text-[10px] text-gray-300 font-mono border border-city-700 flex items-center gap-2 shadow-md">
              <span className="w-2 h-2 rounded-full bg-emerald-500"></span> DIGITAL TWIN SYNC
            </div>
            <DigitalTwinMap 
                focusedCenter={focusedCenter} 
                focusedZoom={selectedIncident ? 18 : null} 
                selectedIncidentId={selectedIncident?.id || null} 
            />
          </div>

          {selectedIncident ? (
            <>
              <IncidentDetailsPanel incident={selectedIncident} />
              <FootagePanel incident={selectedIncident} />

              <div className="grid grid-cols-3 gap-4">
                {DEPARTMENTS.map((dept) => (
                  <DepartmentCard key={dept.id} dept={dept} />
                ))}
              </div>

              <div className="glass-panel flex flex-col border-indigo-500/30 shadow-[0_0_20px_rgba(99,102,241,0.08)] overflow-hidden h-[450px] shrink-0">
                <div className="border-b border-indigo-500/20 p-3.5 text-xs font-bold text-indigo-300 uppercase tracking-widest bg-indigo-900/10 rounded-t-xl flex items-center justify-between shadow-sm">
                  <span className="flex items-center gap-2"><IconBolt className="w-4 h-4" /> Decision Intelligence</span>
                  <span className="text-[9px] bg-indigo-500/20 px-2 py-1 rounded text-indigo-200 border border-indigo-500/30">MULTI-AGENT FUSION</span>
                </div>
                <div className="flex-grow overflow-y-auto">
                  <DecisionCommander incident={selectedIncident} />
                </div>
              </div>

              <div className="glass-panel flex flex-col overflow-hidden h-[380px] shrink-0">
                <div className="border-b border-city-700/50 p-3.5 text-xs font-bold text-gray-300 uppercase tracking-widest bg-city-800/30 rounded-t-xl flex items-center justify-between">
                  <span>City Assistant</span>
                  <span className="text-[9px] bg-blue-500/20 px-2 py-1 rounded text-blue-300 border border-blue-500/30">GROK NLP</span>
                </div>
                <div className="flex-grow overflow-y-auto">
                  <LLMAssistant incident={selectedIncident} />
                </div>
              </div>
            </>
          ) : (
            <div className="glass-panel p-10 flex flex-col items-center justify-center text-gray-500 h-[400px] shrink-0 border-city-700/50 shadow-lg">
              <IconTarget className="w-16 h-16 mb-4 text-city-600" />
              <h2 className="text-xl font-bold text-gray-400 uppercase tracking-widest">Incident Not Found</h2>
              <p className="text-sm mt-2 max-w-md text-center">Please select an active incident from the sidebar to view details, live footage, and AI multi-agent decision intelligence.</p>
            </div>
          )}
        </section>

        <section className="col-span-2 h-full min-h-0 flex flex-col">
          <IncidentsSidebar incidents={activeIncidents} selectedId={selectedId} onSelect={handleSelectIncident} tall />
        </section>
      </main>
    </div>
  );
}