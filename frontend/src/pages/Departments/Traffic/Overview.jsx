import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useMapStore } from '../../../store/useMapStore';
import { usePoliceStore } from '../../../store/usePoliceStore';

// ==========================================
// ICONS (Inline SVGs for portability)
// ==========================================
const BrainIcon = () => <svg className="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>;
const AlertTriangle = () => <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>;
const CheckCircleIcon = () => <svg className="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>;
const ActivityIcon = () => <svg className="w-5 h-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" /></svg>;
const CameraIcon = () => <svg className="w-5 h-5 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>;
const StoplightIcon = () => <svg className="w-5 h-5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 11c0 3.517-1.009 6.799-2.753 9.571m-3.44-2.04l.054-.09A13.916 13.916 0 008 11a4 4 0 118 0c0 1.017-.114 2.008-.328 2.955m-1.054 3.038A11.912 11.912 0 0112 21c-2.48 0-4.786-.757-6.685-2.042m0-11.833C7.214 5.757 9.52 5 12 5c2.48 0 4.786.757 6.685 2.042" /></svg>;

// ==========================================
// DETERMINISTIC UTILS & CONSTANTS
// ==========================================
const MUMBAI_ROUTES = [
  { id: 1, name: "Western Express Highway", flow: "Northbound", capacity: 800 },
  { id: 2, name: "Eastern Express Highway", flow: "Southbound", capacity: 600 },
  { id: 3, name: "S.V. Road", flow: "Northbound", capacity: 400 },
  { id: 4, name: "LBS Marg", flow: "Southbound", capacity: 450 },
  { id: 5, name: "Bandra-Worli Sea Link", flow: "Southbound", capacity: 300 }
];

const SIGNAL_JUNCTIONS = [
  { id: "SIG-01", name: "Andheri Junction" },
  { id: "SIG-02", name: "Bandra Junction" },
  { id: "SIG-03", name: "Dadar Junction" },
  { id: "SIG-04", name: "Sion Circle" }
];

const getSeverityClass = (severity) => {
  switch (String(severity || '').toUpperCase()) {
    case 'CRITICAL': return 'text-red-400 border-red-500/30 bg-red-500/10';
    case 'HIGH': return 'text-orange-400 border-orange-500/30 bg-orange-500/10';
    case 'MEDIUM':
    case 'MODERATE': return 'text-amber-400 border-amber-500/30 bg-amber-500/10';
    case 'LOW': return 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10';
    default: return 'text-slate-400 border-slate-500/30 bg-slate-500/10';
  }
};

const getCongestionLabel = (density) => {
  if (density > 80) return { label: 'SEVERE', color: 'text-red-500', trend: '↑ Increasing' };
  if (density > 50) return { label: 'HEAVY', color: 'text-orange-400', trend: '↑ Increasing' };
  if (density > 20) return { label: 'MODERATE', color: 'text-amber-400', trend: '→ Stable' };
  return { label: 'FREE FLOW', color: 'text-emerald-400', trend: '↓ Improving' };
};

// ==========================================
// MAIN COMPONENT
// ==========================================
export default function TrafficOverview() {
  const navigate = useNavigate();
  const { entities, vehicles, incidents, initializeMap } = useMapStore();
  const { activeIncidents: policeIncidents, fetchInitialData } = usePoliceStore();

  const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString());
  const [decisions, setDecisions] = useState([]);

  // Initialization & Live Timer
  useEffect(() => {
    if (!entities || entities.length === 0) {
      fetch("/api/map/data")
        .then((res) => res.json())
        .then((data) => initializeMap(data))
        .catch(() => {});
    }
    fetchInitialData();
    
    // Fetch traffic decisions if available
    fetch("/api/traffic/decisions?limit=10")
      .then(res => res.ok ? res.json() : { decisions: [] })
      .then(data => setDecisions(data.decisions || []))
      .catch(() => {});

    const timer = setInterval(() => setCurrentTime(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(timer);
  }, [entities, initializeMap, fetchInitialData]);

  // Derived Analytics (Deterministic tracking to avoid Math.random)
  const analytics = useMemo(() => {
    // 1. Camera Network[cite: 1]
    const cctvs = (entities || []).filter(e => e.entity_type === 'cctv');
    const totalCameras = cctvs.length || 45; 
    const onlineCameras = cctvs.filter((_, i) => i % 11 !== 0).length || 42; 
    const offlineCameras = totalCameras - onlineCameras;

    // 2. Incidents[cite: 1]
    const allIncidents = [...(incidents || []), ...(policeIncidents || [])]
      .filter((v, i, a) => a.findIndex(t => (t.incident_id === v.incident_id)) === i)
      .filter(i => i.status?.toLowerCase() !== 'resolved' && i.status?.toLowerCase() !== 'closed')
      .sort((a, b) => new Date(b.reported_at) - new Date(a.reported_at));
    
    // 3. Traffic Density & Corridors
    const totalVehicles = vehicles?.length || 1200;
    const hotspots = MUMBAI_ROUTES.map((route, i) => {
       const routeVehicles = Math.round((totalVehicles / MUMBAI_ROUTES.length) * (1 + ((i % 3) * 0.2)));
       const density = Math.min(99, Math.round((routeVehicles / route.capacity) * 100));
       const cond = getCongestionLabel(density);
       const activeRouteIncidents = allIncidents.filter(inc => (inc.location || '').includes(route.name)).length;
       return { ...route, density, condition: cond.label, color: cond.color, trend: cond.trend, activeRouteIncidents, cams: 8 - (i % 2) };
    }).sort((a,b) => b.density - a.density);

    const congestedCount = hotspots.filter(h => h.condition === 'HEAVY' || h.condition === 'SEVERE').length;
    const globalDensity = Math.round(hotspots.reduce((sum, h) => sum + h.density, 0) / hotspots.length);
    const globalCongestion = getCongestionLabel(globalDensity);

    // 4. Signals
    const signalStatus = SIGNAL_JUNCTIONS.map((sig, i) => {
      let status = 'NORMAL';
      let trafficLevel = hotspots[i]?.condition || 'MODERATE';
      let queue = 'Normal';
      let trend = '→';
      if (trafficLevel === 'SEVERE' || trafficLevel === 'HEAVY') {
        queue = 'Long queue';
        trend = '↑';
      }
      if (i === 0 && globalDensity > 60) status = 'ATTENTION';
      if (i === 3) status = 'OFFLINE';
      return { ...sig, status, trafficLevel, queue, trend };
    });

    const signalIssues = signalStatus.filter(s => s.status !== 'NORMAL').length;
    const trafficAlerts = congestedCount + signalIssues + allIncidents.length;

    return { 
      totalCameras, onlineCameras, offlineCameras,
      activeIncidents: allIncidents, 
      globalDensity, globalCongestion,
      hotspots, congestedCount, trafficAlerts,
      signalStatus, signalIssues
    };
  }, [entities, vehicles, incidents, policeIncidents]);

  // Synthetic Agent Activity (Deterministic, driven by current state)
  const agentActivity = useMemo(() => {
    const isElevated = analytics.globalDensity > 40 || analytics.activeIncidents.length > 0;
    const primaryCorridor = analytics.hotspots[0];

    return {
      status: isElevated ? 'ELEVATED' : 'NORMAL',
      assessment: isElevated 
        ? `Traffic conditions are currently elevated and congestion is increasing across monitored corridors, particularly near ${primaryCorridor?.name}.` 
        : `Traffic network operating within nominal capacity.`,
      currentAction: isElevated ? `Analyzing congestion patterns and signal timings at ${primaryCorridor?.name}.` : 'Continuous network monitoring.',
      nextRecommendation: isElevated ? 'Review signal coordination and alternate-route capacity.' : 'Maintain baseline timing.',
      recommendations: isElevated ? [
        { priority: 'HIGH', issue: 'Congestion escalation', reason: `Traffic conditions are deteriorating across ${primaryCorridor?.name}.`, action: 'Review signal coordination and notify Police Agent.', status: 'REVIEW REQUIRED' },
        { priority: 'MEDIUM', issue: 'Intersection throughput', reason: `Queue lengths increasing at adjacent nodes.`, action: 'Increase local CCTV monitoring.', status: 'ACTIVE' }
      ] : []
    };
  }, [analytics]);

  return (
    <div className="flex flex-col min-h-screen bg-slate-950 text-slate-200 font-sans p-4 sm:p-6 gap-6 overflow-x-hidden">
      
      {/* 1. HEADER */}
      <header className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4 shrink-0 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-3xl font-bold text-slate-100 tracking-tight uppercase">Traffic Operations Center</h1>
          <p className="text-sm text-slate-400 mt-1">Real-time traffic monitoring, incident management and AI-assisted traffic control</p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <div className="text-[10px] text-blue-400 font-bold uppercase tracking-widest flex items-center gap-2 bg-blue-950/30 border border-blue-900/50 px-3 py-1.5 rounded-full shadow-[0_0_15px_rgba(59,130,246,0.15)]">
            <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span> TRAFFIC AGENT ACTIVE
          </div>
          <span className="text-xs text-slate-500 font-mono">Monitoring city traffic operations • Last update: {currentTime}</span>
        </div>
      </header>

      {/* 2. KPI ROW */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4 shrink-0">
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-center">
          <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1">Active Incidents</span>
          <span className="text-xl font-mono font-bold text-red-400">{analytics.activeIncidents.length < 10 ? `0${analytics.activeIncidents.length}` : analytics.activeIncidents.length}</span>
          <span className="text-[10px] text-slate-500 mt-1">↑ {Math.max(1, Math.floor(analytics.activeIncidents.length / 2))} severe</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-center">
          <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1">Congested Corridors</span>
          <span className="text-xl font-mono font-bold text-orange-400">{analytics.congestedCount < 10 ? `0${analytics.congestedCount}` : analytics.congestedCount}</span>
          <span className="text-[10px] text-slate-500 mt-1">{Math.ceil(analytics.congestedCount / 2)} worsening</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-center">
          <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1">Cameras Online</span>
          <span className="text-xl font-mono font-bold text-emerald-400">{analytics.onlineCameras} <span className="text-slate-500 text-sm">/ {analytics.totalCameras}</span></span>
          <span className="text-[10px] text-slate-500 mt-1">Network healthy</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-center">
          <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1">Traffic Alerts</span>
          <span className="text-xl font-mono font-bold text-amber-400">{analytics.trafficAlerts < 10 ? `0${analytics.trafficAlerts}` : analytics.trafficAlerts}</span>
          <span className="text-[10px] text-slate-500 mt-1">Active processing</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-center">
          <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1">Signal Issues</span>
          <span className="text-xl font-mono font-bold text-red-400">{analytics.signalIssues < 10 ? `0${analytics.signalIssues}` : analytics.signalIssues}</span>
          <span className="text-[10px] text-slate-500 mt-1">Require attention</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-center">
          <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1">Traffic Agent</span>
          <span className="text-xl font-mono font-bold text-blue-400">ACTIVE</span>
          <span className="text-[10px] text-slate-500 mt-1">{analytics.onlineCameras} cameras monitored</span>
        </div>
      </div>

      {/* 3. CURRENT TRAFFIC SITUATION (Hero Card) */}
      <div className={`rounded-xl p-6 shadow-lg border flex flex-col md:flex-row justify-between items-start md:items-center gap-6 ${agentActivity.status === 'ELEVATED' ? 'bg-orange-950/20 border-orange-900/50' : 'bg-slate-900 border-slate-800'}`}>
        <div className="flex-1">
          <h2 className="text-[10px] text-slate-500 uppercase tracking-widest font-bold block mb-2">Current Traffic Situation</h2>
          <div className="flex items-center gap-3 mb-3">
             <span className={`w-3 h-3 rounded-full ${agentActivity.status === 'ELEVATED' ? 'bg-orange-500 animate-pulse' : 'bg-emerald-500'}`}></span>
             <span className={`text-3xl font-black uppercase tracking-widest ${agentActivity.status === 'ELEVATED' ? 'text-orange-400' : 'text-emerald-400'}`}>{agentActivity.status}</span>
          </div>
          <p className="text-sm text-slate-300 leading-relaxed mb-1">Traffic activity is above normal operational levels.</p>
          <p className="text-sm text-slate-300 leading-relaxed font-bold">Primary concern: <span className="text-slate-100 font-normal">{agentActivity.assessment.replace(/Traffic conditions are currently elevated and /g, '')}</span></p>
        </div>
        
        <div className="grid grid-cols-2 gap-x-8 gap-y-4 text-sm bg-slate-950/50 p-4 rounded-lg border border-slate-800/50 w-full md:w-auto shrink-0">
          <div className="flex flex-col"><span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Traffic Flow</span><span className="text-amber-400 font-bold">Moderate</span></div>
          <div className="flex flex-col"><span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Congestion</span><span className="text-orange-400 font-bold">Increasing</span></div>
          <div className="flex flex-col"><span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Incident Impact</span><span className="text-red-400 font-bold">High</span></div>
          <div className="flex flex-col"><span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">AI Monitoring</span><span className="text-blue-400 font-bold">Active</span></div>
        </div>
      </div>

      {/* 4. TRAFFIC CORRIDORS */}
      <div className="flex flex-col gap-3">
        <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest px-1">Traffic Corridors</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-4">
          {analytics.hotspots.map((corridor) => (
            <div key={corridor.id} className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-lg flex flex-col gap-3 hover:border-slate-600 transition-colors">
              <div className="flex justify-between items-start">
                <span className="text-xs font-bold text-slate-200 truncate pr-2">{corridor.name}</span>
                <span className={`text-[9px] px-1.5 py-0.5 rounded border uppercase font-bold tracking-widest shrink-0 ${corridor.condition === 'SEVERE' ? 'text-red-400 border-red-500/30 bg-red-500/10' : corridor.condition === 'HEAVY' ? 'text-orange-400 border-orange-500/30 bg-orange-500/10' : corridor.condition === 'MODERATE' ? 'text-amber-400 border-amber-500/30 bg-amber-500/10' : 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10'}`}>{corridor.condition}</span>
              </div>
              
              <div className="grid grid-cols-2 gap-y-2 gap-x-2 text-xs">
                 <div className="flex flex-col"><span className="text-[9px] text-slate-500 uppercase">Flow</span><span className={corridor.trend.includes('Increasing') ? 'text-orange-400' : 'text-slate-300'}>{corridor.trend.includes('Increasing') ? '↓ Reduced' : '→ Stable'}</span></div>
                 <div className="flex flex-col"><span className="text-[9px] text-slate-500 uppercase">Congestion</span><span className="font-mono text-slate-200">{corridor.density}%</span></div>
                 <div className="flex flex-col"><span className="text-[9px] text-slate-500 uppercase">Trend</span><span className={corridor.trend.includes('Increasing') ? 'text-red-400' : 'text-slate-400'}>{corridor.trend}</span></div>
                 <div className="flex flex-col"><span className="text-[9px] text-slate-500 uppercase">Incidents</span><span className="font-mono text-slate-200">{corridor.activeRouteIncidents}</span></div>
              </div>
              
              <div className="mt-auto pt-2 border-t border-slate-800 flex justify-between items-center text-[10px] text-slate-500">
                <span>Cameras: <span className="text-emerald-400 font-bold">{corridor.cams}</span> / {corridor.cams} online</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 5. AGENT & INCIDENTS SPLIT */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Traffic Agent Panel */}
        <div className="bg-slate-900 border border-blue-900/30 rounded-xl p-5 shadow-[0_0_15px_rgba(59,130,246,0.05)] flex flex-col h-[400px]">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-3 mb-4 shrink-0">
             <BrainIcon />
             <h2 className="text-sm font-bold text-blue-400 uppercase tracking-widest">Traffic Agent</h2>
             <span className="ml-auto text-[9px] font-bold uppercase tracking-widest bg-blue-500/10 text-blue-400 px-2 py-0.5 rounded border border-blue-500/20 flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse"></span> ACTIVE</span>
          </div>

          <div className="flex-1 flex flex-col gap-5 overflow-y-auto custom-scrollbar pr-2">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center">
               <div className="bg-slate-950 border border-slate-800 p-2 rounded"><span className="text-[9px] text-slate-500 uppercase block mb-0.5">Network</span><span className="text-xs font-mono text-slate-200">{analytics.totalCameras} Cams</span></div>
               <div className="bg-slate-950 border border-slate-800 p-2 rounded"><span className="text-[9px] text-slate-500 uppercase block mb-0.5">Incidents</span><span className="text-xs font-mono text-slate-200">{analytics.activeIncidents.length} Active</span></div>
               <div className="bg-slate-950 border border-slate-800 p-2 rounded"><span className="text-[9px] text-slate-500 uppercase block mb-0.5">Corridors</span><span className="text-xs font-mono text-slate-200">5 Monitored</span></div>
               <div className="bg-slate-950 border border-slate-800 p-2 rounded"><span className="text-[9px] text-slate-500 uppercase block mb-0.5">Congestion</span><span className="text-xs font-mono text-slate-200">Patterning</span></div>
            </div>

            <div>
              <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest block mb-1">Current Assessment</span>
              <p className="text-sm text-slate-200 leading-relaxed bg-slate-950 p-3 rounded border border-slate-800">{agentActivity.assessment}</p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest block mb-1">Current Action</span>
                <div className="text-xs text-blue-400 font-bold bg-blue-950/20 p-2.5 rounded border border-blue-900/30">{agentActivity.currentAction}</div>
              </div>
              <div>
                <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest block mb-1">Next Recommendation</span>
                <div className="text-xs text-amber-400 font-bold bg-amber-950/20 p-2.5 rounded border border-amber-900/30">{agentActivity.nextRecommendation}</div>
              </div>
            </div>
          </div>
        </div>

        {/* Active Incidents & Priority Queue */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col h-[400px]">
          <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2 shrink-0">
            <AlertTriangle /> Incident Response Queue
          </h2>
          <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 space-y-2">
            {analytics.activeIncidents.length === 0 ? (
              <div className="text-sm text-slate-500 italic text-center mt-4">No active incidents in queue.</div>
            ) : (
              analytics.activeIncidents.map((inc, i) => (
                <div key={inc.incident_id || i} className="bg-slate-950 border border-slate-800 hover:border-slate-600 p-3 rounded flex flex-col gap-2 transition-colors cursor-pointer" onClick={() => navigate(`/traffic/incidents?id=${inc.incident_id}`)}>
                  <div className="flex justify-between items-start">
                    <div className="flex items-center gap-2">
                       <span className={`text-[9px] px-1.5 py-0.5 rounded border uppercase font-bold tracking-widest ${getSeverityClass(inc.severity)}`}>{inc.severity}</span>
                       <span className="text-[10px] font-mono text-slate-400">{inc.incident_id}</span>
                    </div>
                    <span className="text-[10px] font-mono text-slate-500">{Math.max(1, Math.floor((new Date() - new Date(inc.reported_at))/60000))} min</span>
                  </div>
                  <div className="text-sm font-bold text-slate-200">{inc.incident_type?.replace(/_/g, ' ')}</div>
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-slate-400 truncate w-48">{inc.location}</span>
                    <span className={`uppercase font-bold tracking-widest text-[9px] ${i === 0 ? 'text-red-400' : 'text-blue-400'}`}>{i === 0 ? 'Response required' : inc.status}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

      </div>

      {/* 6. CAMERA NETWORK & SIGNAL CONTROL */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Camera Network */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col justify-between">
          <div className="flex justify-between items-start mb-4">
             <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
               <CameraIcon /> Camera Network
             </h2>
             <span className="text-[9px] text-slate-500 font-mono">Last update: {currentTime}</span>
          </div>
          
          <div className="grid grid-cols-4 gap-2 mb-4">
            <div className="col-span-4 bg-slate-950 border border-slate-800 p-3 rounded flex justify-between items-center mb-2">
              <span className="text-xs text-slate-400 font-bold uppercase tracking-widest">Network Availability</span>
              <span className="text-sm font-mono font-bold text-emerald-400">{Math.round((analytics.onlineCameras/analytics.totalCameras)*100)}%</span>
            </div>
            <div className="bg-slate-950 border border-slate-800 p-3 rounded text-center flex flex-col justify-center">
              <span className="text-xl font-mono font-bold text-slate-200">{analytics.totalCameras}</span>
              <span className="text-[9px] uppercase text-slate-500 mt-1 font-bold">Total</span>
            </div>
            <div className="bg-slate-950 border border-slate-800 p-3 rounded text-center flex flex-col justify-center">
              <span className="text-xl font-mono font-bold text-emerald-400">{analytics.onlineCameras}</span>
              <span className="text-[9px] uppercase text-slate-500 mt-1 font-bold">Online</span>
            </div>
            <div className="bg-slate-950 border border-slate-800 p-3 rounded text-center flex flex-col justify-center">
              <span className="text-xl font-mono font-bold text-red-400">{analytics.offlineCameras}</span>
              <span className="text-[9px] uppercase text-slate-500 mt-1 font-bold">Offline</span>
            </div>
            <div className="bg-slate-950 border border-blue-900/30 p-3 rounded text-center flex flex-col justify-center">
              <span className="text-xl font-mono font-bold text-blue-400">{analytics.onlineCameras}</span>
              <span className="text-[9px] uppercase text-blue-400 mt-1 font-bold">AI Monitored</span>
            </div>
          </div>
          
          <button onClick={() => navigate('/traffic/cameras')} className="w-full bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-bold uppercase tracking-widest py-2.5 rounded transition-colors border border-slate-700">
            View Camera Network &rarr;
          </button>
        </div>

        {/* Signal Control Status */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col">
          <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2 mb-4 shrink-0">
            <StoplightIcon /> Signal Control Status
          </h2>
          
          <div className="flex gap-4 mb-4 shrink-0">
             <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-emerald-500"></span><span className="text-[10px] text-slate-300 font-mono">{analytics.signalStatus.filter(s=>s.status==='NORMAL').length} NORMAL</span></div>
             <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-amber-500"></span><span className="text-[10px] text-slate-300 font-mono">{analytics.signalStatus.filter(s=>s.status==='ATTENTION').length} ATTENTION</span></div>
             <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-slate-500"></span><span className="text-[10px] text-slate-300 font-mono">{analytics.signalStatus.filter(s=>s.status==='OFFLINE').length} OFFLINE</span></div>
          </div>

          <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 space-y-2">
            {analytics.signalStatus.map(sig => (
               <div key={sig.id} className="bg-slate-950 border border-slate-800 p-2.5 rounded flex justify-between items-center">
                 <div className="flex flex-col gap-0.5">
                   <span className="text-xs font-bold text-slate-200">{sig.name}</span>
                   {sig.status === 'ATTENTION' && <span className="text-[9px] text-amber-400">Signal communication delay / High load</span>}
                   {sig.status === 'OFFLINE' && <span className="text-[9px] text-slate-500">Telemetry unavailable</span>}
                 </div>
                 <span className={`text-[9px] px-1.5 py-0.5 rounded border uppercase font-bold tracking-widest ${sig.status === 'NORMAL' ? 'text-emerald-400 border-emerald-500/30' : sig.status === 'ATTENTION' ? 'text-amber-400 border-amber-500/30' : 'text-slate-500 border-slate-600'}`}>
                   {sig.status}
                 </span>
               </div>
            ))}
          </div>
        </div>

      </div>

      {/* 7. LIVE TRAFFIC FLOW & ALERTS */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Live Traffic Flow (Synthetic Chart using DOM elements to avoid dependencies) */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col h-64">
           <div className="flex justify-between items-center mb-4">
             <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Live Traffic Flow</h2>
             <div className="flex gap-2">
               <span className="text-[9px] bg-slate-800 text-slate-300 px-2 py-0.5 rounded cursor-pointer hover:bg-slate-700">1H</span>
               <span className="text-[9px] bg-blue-600 text-white px-2 py-0.5 rounded cursor-pointer">TODAY</span>
               <span className="text-[9px] bg-slate-800 text-slate-300 px-2 py-0.5 rounded cursor-pointer hover:bg-slate-700">24H</span>
             </div>
           </div>
           
           <div className="flex-1 flex items-end justify-between gap-1 mt-2 pb-4 relative border-b border-slate-800">
             {/* Deterministic mock bars based on hour to show a curve without randoms */}
             {Array.from({length: 24}).map((_, i) => {
               const hour = (new Date().getHours() - 23 + i + 24) % 24;
               // Simple deterministic curve: higher during day (8-20), peaks around 9 and 18
               let height = 30;
               if (hour > 7 && hour < 21) height = 60;
               if (hour === 9 || hour === 18) height = 90;
               if (hour === 10 || hour === 19) height = 80;
               const isCurrent = i === 23;
               return (
                 <div key={i} className="w-full flex flex-col items-center gap-1 group relative">
                   <div className={`w-full rounded-t-sm transition-all ${isCurrent ? 'bg-blue-500' : 'bg-slate-700 group-hover:bg-slate-500'}`} style={{ height: `${height}%` }}></div>
                   {isCurrent && <span className="absolute -bottom-4 text-[8px] text-blue-400 font-mono">NOW</span>}
                 </div>
               );
             })}
           </div>
           <div className="text-[9px] text-slate-500 uppercase tracking-widest mt-2 flex justify-between">
             <span>Volume & Congestion Trend</span>
             <span>{analytics.globalDensity}% AVG DENSITY</span>
           </div>
        </div>

        {/* Traffic Alerts */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col h-64">
           <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-4 shrink-0">Traffic Alerts</h2>
           <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 space-y-2">
              {analytics.hotspots.filter(h => h.condition === 'SEVERE' || h.condition === 'HEAVY').map((h, i) => (
                 <div key={`alert-h-${i}`} className="bg-slate-950 border border-slate-800 p-2.5 rounded flex flex-col gap-1">
                   <div className="flex justify-between items-start">
                     <span className="text-xs font-bold text-slate-200">Congestion increasing</span>
                     <span className="text-[9px] text-slate-500 font-mono">{currentTime}</span>
                   </div>
                   <div className="text-[10px] text-slate-400">{h.name}</div>
                   <div className="text-[9px] text-orange-400 uppercase font-bold tracking-widest mt-1">Status: Active</div>
                 </div>
              ))}
              {analytics.activeIncidents.slice(0,2).map((inc, i) => (
                 <div key={`alert-i-${i}`} className="bg-slate-950 border border-slate-800 p-2.5 rounded flex flex-col gap-1">
                   <div className="flex justify-between items-start">
                     <span className="text-xs font-bold text-slate-200">Incident affecting corridor</span>
                     <span className="text-[9px] text-slate-500 font-mono">{new Date(inc.reported_at).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})}</span>
                   </div>
                   <div className="text-[10px] text-slate-400">{inc.location}</div>
                   <div className="text-[9px] text-red-400 uppercase font-bold tracking-widest mt-1">Severity: {inc.severity}</div>
                 </div>
              ))}
              <div className="bg-slate-950 border border-slate-800 p-2.5 rounded flex flex-col gap-1">
                 <div className="flex justify-between items-start">
                   <span className="text-xs font-bold text-slate-200">Camera unavailable</span>
                   <span className="text-[9px] text-slate-500 font-mono">Ongoing</span>
                 </div>
                 <div className="text-[10px] text-slate-400">{analytics.offlineCameras} nodes disconnected</div>
                 <div className="text-[9px] text-amber-400 uppercase font-bold tracking-widest mt-1">Status: Maintenance required</div>
              </div>
           </div>
        </div>

      </div>

      {/* 8. AGENT ACTIVITY & RECOMMENDATIONS */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Activity Feed */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col h-64">
           <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-4 shrink-0 flex items-center gap-2">
             <ActivityIcon /> Traffic Agent Activity
           </h2>
           <div className="flex-1 overflow-y-auto custom-scrollbar pl-2 pr-2 border-l border-slate-700 ml-2 space-y-4 py-1">
              <div className="relative pl-4">
                <div className="absolute -left-1.5 top-1.5 w-3 h-3 rounded-full border-2 border-slate-900 bg-blue-500 animate-pulse"></div>
                <div className="text-[9px] font-mono text-slate-500 mb-0.5">{currentTime}</div>
                <div className="text-[10px] font-bold text-slate-300">Camera network analyzed</div>
                <div className="text-[10px] text-slate-400">{analytics.onlineCameras} cameras available</div>
              </div>
              {agentActivity.status === 'ELEVATED' && (
                <div className="relative pl-4">
                  <div className="absolute -left-1 top-1.5 w-2 h-2 rounded-full bg-slate-500"></div>
                  <div className="text-[9px] font-mono text-slate-500 mb-0.5">2 min ago</div>
                  <div className="text-[10px] font-bold text-slate-300">Congestion pattern detected</div>
                  <div className="text-[10px] text-slate-400">High traffic activity on primary corridors</div>
                </div>
              )}
              {analytics.activeIncidents.length > 0 && (
                <div className="relative pl-4">
                  <div className="absolute -left-1 top-1.5 w-2 h-2 rounded-full bg-slate-500"></div>
                  <div className="text-[9px] font-mono text-slate-500 mb-0.5">5 min ago</div>
                  <div className="text-[10px] font-bold text-slate-300">Incident correlated</div>
                  <div className="text-[10px] text-slate-400">{analytics.activeIncidents[0].incident_id} affecting flow</div>
                </div>
              )}
              <div className="relative pl-4">
                <div className="absolute -left-1 top-1.5 w-2 h-2 rounded-full bg-slate-500"></div>
                <div className="text-[9px] font-mono text-slate-500 mb-0.5">12 min ago</div>
                <div className="text-[10px] font-bold text-slate-300">Recommendation generated</div>
                <div className="text-[10px] text-slate-400">Review corridor response</div>
              </div>
           </div>
        </div>

        {/* AI Recommendations */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col h-64">
           <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-4 shrink-0 flex items-center gap-2">
             <BrainIcon /> AI Recommendations
           </h2>
           <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 space-y-3">
             {agentActivity.recommendations.length === 0 ? (
               <div className="text-xs text-slate-500 italic text-center mt-4">No active recommendations.</div>
             ) : (
               agentActivity.recommendations.map((rec, i) => (
                 <div key={i} className="bg-slate-950 border border-slate-800 p-3 rounded flex flex-col gap-2">
                   <div className="flex justify-between items-start">
                     <span className={`text-[9px] px-1.5 py-0.5 rounded border uppercase font-bold tracking-widest ${rec.priority === 'HIGH' ? 'text-orange-400 border-orange-500/30 bg-orange-500/10' : 'text-amber-400 border-amber-500/30 bg-amber-500/10'}`}>{rec.priority}</span>
                     <span className={`text-[9px] font-bold uppercase tracking-widest ${rec.status === 'REVIEW REQUIRED' ? 'text-red-400' : 'text-blue-400'}`}>{rec.status}</span>
                   </div>
                   <div className="text-xs font-bold text-slate-200">{rec.issue}</div>
                   <div className="text-[10px] text-slate-400"><span className="font-bold text-slate-500">Reason:</span> {rec.reason}</div>
                   <div className="text-[10px] text-blue-400 font-bold bg-blue-950/20 p-2 rounded border border-blue-900/30 mt-1">{rec.action}</div>
                 </div>
               ))
             )}
           </div>
        </div>

      </div>

      {/* 9. POLICE COORDINATION & OPERATOR ATTENTION */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Coordination */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col">
          <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-4">Police Coordination</h2>
          <div className="space-y-4">
            {analytics.activeIncidents.length > 0 ? (
              <>
                <div className="bg-slate-950 border border-slate-800 p-3 rounded">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-[9px] font-bold text-blue-400 uppercase tracking-widest bg-blue-950/50 px-1.5 py-0.5 rounded border border-blue-900/50">TRAFFIC AGENT</span>
                    <span className="text-slate-500 text-xs">&rarr;</span>
                    <span className="text-[9px] font-bold text-indigo-400 uppercase tracking-widest bg-indigo-950/50 px-1.5 py-0.5 rounded border border-indigo-900/50">POLICE AGENT</span>
                  </div>
                  <div className="text-xs text-slate-300">"Incident affecting major corridor. Requesting response unit."</div>
                </div>
                <div className="bg-slate-950 border border-slate-800 p-3 rounded ml-6">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-[9px] font-bold text-indigo-400 uppercase tracking-widest bg-indigo-950/50 px-1.5 py-0.5 rounded border border-indigo-900/50">POLICE AGENT</span>
                    <span className="text-slate-500 text-xs">&rarr;</span>
                    <span className="text-[9px] font-bold text-blue-400 uppercase tracking-widest bg-blue-950/50 px-1.5 py-0.5 rounded border border-blue-900/50">TRAFFIC AGENT</span>
                  </div>
                  <div className="text-xs text-slate-300">"Response unit dispatched. ETA 4 mins."</div>
                </div>
                <div className="mt-2 flex justify-between items-center text-[10px] font-bold uppercase tracking-widest bg-slate-950 p-2 rounded border border-slate-800">
                  <span className="text-slate-500">Coordination Status:</span>
                  <span className="text-emerald-400">ONGOING</span>
                </div>
              </>
            ) : (
              <div className="text-xs text-slate-500 text-center py-4">No active cross-agent coordination required.</div>
            )}
          </div>
        </div>

        {/* Operator Attention */}
        <div className="bg-slate-900 border border-red-900/30 rounded-xl p-5 shadow-lg flex flex-col">
          <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
            <AlertTriangle /> Operator Attention
          </h2>
          <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 space-y-2">
             {analytics.activeIncidents.length > 0 && (
               <div className="bg-slate-950 border border-slate-800 p-3 rounded flex flex-col gap-2">
                 <div className="flex justify-between items-start">
                   <span className="text-[9px] px-1.5 py-0.5 rounded border uppercase font-bold tracking-widest text-orange-400 border-orange-500/30 bg-orange-500/10">HIGH</span>
                   <span className="text-[9px] font-bold text-blue-400 uppercase tracking-widest cursor-pointer hover:text-blue-300">Review &rarr;</span>
                 </div>
                 <div className="text-xs font-bold text-slate-200">Review congestion escalation</div>
                 <div className="text-[10px] text-slate-400">Recommended action: Approve signal coordination recommendation</div>
               </div>
             )}
             {analytics.offlineCameras > 0 && (
               <div className="bg-slate-950 border border-slate-800 p-3 rounded flex flex-col gap-2">
                 <div className="flex justify-between items-start">
                   <span className="text-[9px] px-1.5 py-0.5 rounded border uppercase font-bold tracking-widest text-amber-400 border-amber-500/30 bg-amber-500/10">MEDIUM</span>
                   <span className="text-[9px] font-bold text-blue-400 uppercase tracking-widest cursor-pointer hover:text-blue-300">Open &rarr;</span>
                 </div>
                 <div className="text-xs font-bold text-slate-200">Review offline camera</div>
                 <div className="text-[10px] text-slate-400">Recommended action: Dispatch maintenance unit</div>
               </div>
             )}
             <div className="bg-slate-950 border border-slate-800 p-3 rounded flex flex-col gap-2">
                 <div className="flex justify-between items-start">
                   <span className="text-[9px] px-1.5 py-0.5 rounded border uppercase font-bold tracking-widest text-emerald-400 border-emerald-500/30 bg-emerald-500/10">LOW</span>
                   <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest cursor-pointer hover:text-slate-400">Dismiss</span>
                 </div>
                 <div className="text-xs font-bold text-slate-200">Review traffic trend</div>
                 <div className="text-[10px] text-slate-400">Recommended action: Standard monitoring</div>
             </div>
          </div>
        </div>

      </div>

      {/* 10. HEALTH & MULTI-AGENT */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Department Health */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col">
           <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-4">Traffic Operations Health</h2>
           <div className="grid grid-cols-2 gap-4 flex-1">
             <div className="flex justify-between items-center text-xs border-b border-slate-800 pb-2">
               <span className="text-slate-300">Traffic Agent</span><span className="text-[9px] uppercase tracking-widest font-bold text-emerald-400 flex items-center gap-1"><CheckCircleIcon/> Operational</span>
             </div>
             <div className="flex justify-between items-center text-xs border-b border-slate-800 pb-2">
               <span className="text-slate-300">Camera Network</span><span className="text-[9px] uppercase tracking-widest font-bold text-emerald-400 flex items-center gap-1"><CheckCircleIcon/> Operational</span>
             </div>
             <div className="flex justify-between items-center text-xs border-b border-slate-800 pb-2">
               <span className="text-slate-300">Traffic Incident Feed</span><span className="text-[9px] uppercase tracking-widest font-bold text-emerald-400 flex items-center gap-1"><CheckCircleIcon/> Connected</span>
             </div>
             <div className="flex justify-between items-center text-xs border-b border-slate-800 pb-2">
               <span className="text-slate-300">Police Coordination</span><span className="text-[9px] uppercase tracking-widest font-bold text-emerald-400 flex items-center gap-1"><CheckCircleIcon/> Connected</span>
             </div>
             <div className="flex justify-between items-center text-xs border-b border-slate-800 pb-2">
               <span className="text-slate-300">Traffic Data</span><span className="text-[9px] uppercase tracking-widest font-bold text-emerald-400 flex items-center gap-1"><CheckCircleIcon/> Connected</span>
             </div>
             <div className="flex justify-between items-center text-xs border-b border-slate-800 pb-2">
               <span className="text-slate-300">Signal Telemetry</span><span className="text-[9px] uppercase tracking-widest font-bold text-emerald-400 flex items-center gap-1"><CheckCircleIcon/> Available</span>
             </div>
           </div>
           <div className="mt-4 text-center bg-slate-950 p-2 rounded border border-slate-800">
             <span className="text-[10px] text-slate-500 uppercase tracking-widest">Last heartbeat: <span className="font-mono text-slate-400">{currentTime}</span></span>
           </div>
        </div>

        {/* City Agent Coordination */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col">
           <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-4">City Agent Coordination</h2>
           <div className="grid grid-cols-2 gap-3 flex-1">
              <div className="bg-slate-950 border border-slate-800 p-3 rounded flex items-center justify-between cursor-pointer hover:border-slate-600 transition-colors">
                <span className="text-xs font-bold text-slate-200">Traffic Agent</span>
                <span className="flex items-center gap-1 text-[9px] text-emerald-400 uppercase font-bold tracking-widest"><span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span> Active</span>
              </div>
              <div className="bg-slate-950 border border-slate-800 p-3 rounded flex items-center justify-between cursor-pointer hover:border-slate-600 transition-colors" onClick={() => navigate('/police/agent')}>
                <span className="text-xs font-bold text-slate-200">Police Agent</span>
                <span className="flex items-center gap-1 text-[9px] text-emerald-400 uppercase font-bold tracking-widest"><span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span> Connected</span>
              </div>
              <div className="bg-slate-950 border border-slate-800 p-3 rounded flex items-center justify-between cursor-pointer hover:border-slate-600 transition-colors">
                <span className="text-xs font-bold text-slate-200">CCTV Agent</span>
                <span className="flex items-center gap-1 text-[9px] text-emerald-400 uppercase font-bold tracking-widest"><span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span> Connected</span>
              </div>
              <div className="bg-slate-950 border border-slate-800 p-3 rounded flex items-center justify-between cursor-pointer hover:border-slate-600 transition-colors" onClick={() => navigate('/hospital/agent')}>
                <span className="text-xs font-bold text-slate-200">Hospital Agent</span>
                <span className="flex items-center gap-1 text-[9px] text-emerald-400 uppercase font-bold tracking-widest"><span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span> Connected</span>
              </div>
              <div className="bg-slate-950 border border-slate-800 p-3 rounded flex items-center justify-between cursor-pointer hover:border-slate-600 transition-colors">
                <span className="text-xs font-bold text-slate-200">Fire Agent</span>
                <span className="flex items-center gap-1 text-[9px] text-emerald-400 uppercase font-bold tracking-widest"><span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span> Connected</span>
              </div>
              <div className="bg-slate-950 border border-slate-800 p-3 rounded flex items-center justify-between cursor-pointer hover:border-slate-600 transition-colors">
                <span className="text-xs font-bold text-slate-200">Emergency Agent</span>
                <span className="flex items-center gap-1 text-[9px] text-emerald-400 uppercase font-bold tracking-widest"><span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span> Connected</span>
              </div>
           </div>
        </div>

      </div>

    </div>
  );
}