import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { usePoliceStore } from '../../../store/usePoliceStore';

// ==========================================
// DETERMINISTIC DEMONSTRATION DATA
// ==========================================
// Used strictly for CCTV simulation features not yet available in the backend
const SYNTHETIC_CAMERAS = [
  { id: 'CAM-021', location: 'Bandra West', status: 'ONLINE', aiMonitoring: 'ACTIVE', lastSeen: 'Live', alerts: 4, risk: 'HIGH', health: 100 },
  { id: 'CAM-087', location: 'Andheri East', status: 'ONLINE', aiMonitoring: 'ACTIVE', lastSeen: 'Live', alerts: 2, risk: 'MEDIUM', health: 100 },
  { id: 'CAM-034', location: 'Kurla', status: 'DEGRADED', aiMonitoring: 'LIMITED', lastSeen: '18:38', alerts: 1, risk: 'LOW', health: 45 },
  { id: 'CAM-112', location: 'Dadar Junction', status: 'ONLINE', aiMonitoring: 'ACTIVE', lastSeen: 'Live', alerts: 7, risk: 'CRITICAL', health: 98 },
  { id: 'CAM-145', location: 'Sion Circle', status: 'OFFLINE', aiMonitoring: 'UNAVAILABLE', lastSeen: '04:12', alerts: 0, risk: 'LOW', health: 0 },
];

const SYNTHETIC_DETECTIONS = [
  { id: 'DET-8812', time: '18:42:17', camera: 'CAM-112', location: 'Dadar Junction', type: 'Suspicious Gathering', confidence: 94, severity: 'CRITICAL', status: 'UNACKNOWLEDGED' },
  { id: 'DET-8811', time: '18:41:32', camera: 'CAM-021', location: 'Bandra West', type: 'Unusual Crowd Density', confidence: 88, severity: 'HIGH', status: 'REVIEWING' },
  { id: 'DET-8810', time: '18:39:05', camera: 'CAM-087', location: 'Andheri East', type: 'Vehicle Stopped in Restricted Zone', confidence: 86, severity: 'MEDIUM', status: 'UNACKNOWLEDGED' },
  { id: 'DET-8809', time: '18:35:44', camera: 'CAM-034', location: 'Kurla', type: 'Unusual Movement', confidence: 72, severity: 'LOW', status: 'ACKNOWLEDGED' },
  { id: 'DET-8808', time: '18:32:10', camera: 'CAM-021', location: 'Bandra West', type: 'Potential Anomaly', confidence: 82, severity: 'MEDIUM', status: 'ACKNOWLEDGED' },
];

const SYNTHETIC_TIMELINE = [
  { time: '18:43:02', event: 'POLICE INCIDENT MATCH CHECK', desc: 'Correlating active incidents with Dadar detections' },
  { time: '18:42:45', event: 'RECOMMENDATION GENERATED', desc: 'Agent recommended patrol review for Bandra West' },
  { time: '18:42:17', event: 'HIGH-RISK DETECTION', desc: 'CAM-112 classified suspicious gathering' },
  { time: '18:41:53', event: 'CROSS-CAMERA CORRELATION', desc: 'Movement tracked across CAM-021 and CAM-022' },
  { time: '18:38:14', event: 'CAMERA HEALTH WARNING', desc: 'CAM-034 signal degradation detected' },
];

const SYNTHETIC_QUEUE = [
  { id: 'TSK-1', priority: 'P1', reason: 'Review high-risk CCTV event', camera: 'CAM-112', time: '18:42', status: 'OPEN' },
  { id: 'TSK-2', priority: 'P1', reason: 'Correlate CCTV event with incident', camera: 'CAM-021', time: '18:41', status: 'OPEN' },
  { id: 'TSK-3', priority: 'P2', reason: 'Review camera health', camera: 'CAM-034', time: '18:38', status: 'OPEN' },
];

// ==========================================
// UTILITIES
// ==========================================
const getSeverityStyle = (severity) => {
  switch (String(severity || '').toUpperCase()) {
    case 'CRITICAL': return 'text-red-400 border-red-500/30 bg-red-500/10';
    case 'HIGH': return 'text-orange-400 border-orange-500/30 bg-orange-500/10';
    case 'MEDIUM': return 'text-amber-400 border-amber-500/30 bg-amber-500/10';
    case 'LOW': return 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10';
    default: return 'text-slate-400 border-slate-500/30 bg-slate-500/10';
  }
};

const getTextColor = (severity) => {
  switch (String(severity || '').toUpperCase()) {
    case 'CRITICAL': return 'text-red-400';
    case 'HIGH': return 'text-orange-400';
    case 'MEDIUM': return 'text-amber-400';
    case 'LOW': return 'text-emerald-400';
    default: return 'text-slate-400';
  }
};

// ==========================================
// MAIN COMPONENT
// ==========================================
export default function CCTVIntelligence() {
  const navigate = useNavigate();
  const { activeIncidents, fetchInitialData } = usePoliceStore();
  const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString());

  useEffect(() => {
    fetchInitialData();
    const timer = setInterval(() => setCurrentTime(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(timer);
  }, [fetchInitialData]);

  // Derived Analytics
  const analytics = useMemo(() => {
    const totalCameras = 128;
    const online = 121;
    const offline = 4;
    const degraded = 3;
    const highRisk = SYNTHETIC_DETECTIONS.filter(d => ['CRITICAL', 'HIGH'].includes(d.severity)).length;
    
    const typeDistribution = {
      'Suspicious Activity': 42,
      'Crowd Anomaly': 28,
      'Vehicle Violation': 18,
      'Restricted Area': 12
    };

    return { totalCameras, online, offline, degraded, highRisk, typeDistribution };
  }, []);

  return (
    <div className="flex flex-col min-h-screen bg-slate-950 text-slate-200 font-sans p-4 sm:p-6 overflow-x-hidden">
      
      {/* HEADER */}
      <header className="flex flex-col sm:flex-row justify-between items-start sm:items-end border-b border-slate-800 pb-4 mb-6 shrink-0 gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 uppercase tracking-widest">CCTV Intelligence</h1>
          <p className="text-xs text-slate-400 font-mono mt-1">AI-powered video monitoring, anomaly detection and incident correlation</p>
        </div>
        <div className="text-left sm:text-right">
          <div className="text-[10px] text-cyan-400 font-bold uppercase tracking-widest flex items-center justify-start sm:justify-end gap-1.5 mb-1">
            <span className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse"></span> AI MONITORING ACTIVE
          </div>
          <div className="text-xs text-slate-500 font-mono flex items-center gap-2">
            Last analysis: {currentTime}
            <button className="px-2 py-0.5 bg-slate-900 border border-slate-700 hover:bg-slate-800 rounded uppercase tracking-widest text-[9px]">Refresh</button>
          </div>
        </div>
      </header>

      {/* KPI ROW */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4 mb-6 shrink-0">
        {[
          { label: 'Total Cameras', val: analytics.totalCameras },
          { label: 'Online', val: analytics.online, color: 'text-emerald-400' },
          { label: 'Offline', val: analytics.offline, color: 'text-red-400' },
          { label: 'Degraded', val: analytics.degraded, color: 'text-amber-400' },
          { label: 'AI Alerts', val: SYNTHETIC_DETECTIONS.length, color: 'text-cyan-400' },
          { label: 'High-Risk Events', val: analytics.highRisk, color: 'text-orange-400' }
        ].map((kpi, i) => (
          <div key={i} className="bg-slate-900 border border-slate-800 p-4 rounded-lg flex flex-col justify-center">
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">{kpi.label}</span>
            <span className={`text-2xl font-mono font-bold ${kpi.color || 'text-slate-200'}`}>{kpi.val}</span>
          </div>
        ))}
      </div>

      {/* AI COMMAND CENTER HERO */}
      <div className="bg-slate-900 border border-cyan-900/50 rounded-lg p-6 shadow-[0_0_30px_rgba(6,182,212,0.05)] mb-6 flex flex-col xl:flex-row gap-6 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none"></div>
        
        {/* Agent Assessment */}
        <div className="xl:w-1/3 border-b xl:border-b-0 xl:border-r border-slate-800 pb-6 xl:pb-0 xl:pr-6 flex flex-col">
          <div className="flex justify-between items-center mb-6">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-500 animate-pulse"></span>
              <h2 className="text-sm font-bold text-cyan-400 uppercase tracking-widest">CCTV Intelligence Agent</h2>
            </div>
            <div className="text-right">
              <span className="text-[10px] text-slate-500 uppercase tracking-widest block">Surveillance Risk</span>
              <span className="text-lg font-black text-orange-400 uppercase font-mono">ELEVATED (72/100)</span>
            </div>
          </div>
          
          <div className="flex-grow space-y-5">
            <div>
              <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold block mb-1">Current Assessment</span>
              <p className="text-sm text-slate-300 leading-relaxed border-l-2 border-cyan-500/50 pl-3">
                Elevated unusual activity detected across multiple camera zones. Spatial density patterns indicate potential coordinated anomalies in central transit sectors.
              </p>
            </div>
            <div>
               <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold block mb-1">Reasoning Evidence</span>
               <ul className="text-xs text-slate-400 space-y-1 font-mono list-disc ml-4">
                 <li>Multiple detections within a 5-minute window</li>
                 <li>One high-confidence anomaly in Dadar Junction</li>
                 <li>Related active police incident proximity matches</li>
               </ul>
            </div>
          </div>
        </div>

        {/* Decisions & Video Feed Placeholder */}
        <div className="xl:w-2/3 flex flex-col lg:flex-row gap-6">
          <div className="flex-1 flex flex-col">
            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">Current AI Decision</h3>
            <div className="bg-slate-950/50 border border-slate-800 rounded p-4 flex-grow flex flex-col gap-3">
              <div>
                <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold block mb-1">Recommended Action</span>
                <span className="text-sm text-cyan-400 font-bold">Increase monitoring of affected camera zones. Review adjacent cameras and correlate detections with active police incidents.</span>
              </div>
              <div className="grid grid-cols-2 gap-4 mt-auto border-t border-slate-800 pt-3">
                <div>
                  <span className="text-[10px] text-slate-500 uppercase block">Confidence</span>
                  <span className="text-sm font-mono text-slate-200">88%</span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 uppercase block">Status</span>
                  <span className="text-sm font-mono text-emerald-400">ACTIVE</span>
                </div>
              </div>
            </div>
          </div>

          <div className="flex-1">
            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">Live Analysis Feed</h3>
            <div className="bg-slate-950 border border-slate-800 rounded h-32 lg:h-full min-h-[160px] flex flex-col items-center justify-center relative overflow-hidden group">
              <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMiIgY3k9IjIiIHI9IjIiIGZpbGw9IiMzMzMiIGZpbGwtb3BhY2l0eT0iMC40Ii8+PC9zdmc+')] opacity-20"></div>
              <div className="w-8 h-8 mb-2 text-slate-600 group-hover:text-cyan-500/50 transition-colors">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
              </div>
              <span className="text-xs font-bold text-slate-500 uppercase tracking-widest relative z-10">SIMULATION FEED</span>
              <span className="text-[10px] font-mono text-slate-600 relative z-10">AI event simulation active</span>
            </div>
          </div>
        </div>
      </div>

      {/* FEED & CORRELATION ROW */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-6">
        
        {/* Main Feed */}
        <div className="lg:col-span-8 bg-slate-900 border border-slate-800 rounded-lg flex flex-col max-h-[450px]">
          <div className="p-4 border-b border-slate-800 shrink-0 flex justify-between items-center">
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest">AI Detection Feed</h2>
            <span className="text-[10px] bg-slate-800 px-2 py-0.5 rounded font-mono text-slate-400">Live Stream</span>
          </div>
          <div className="overflow-y-auto custom-scrollbar p-2 space-y-2 flex-grow">
            {SYNTHETIC_DETECTIONS.map(d => (
              <div key={d.id} className="p-3 bg-slate-950 border border-slate-800 hover:border-slate-600 rounded flex flex-col sm:flex-row sm:items-center justify-between gap-4 transition-colors">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-1">
                    <span className="text-[10px] font-mono text-slate-500">{d.time}</span>
                    <span className="text-[10px] font-bold text-slate-300 bg-slate-800 px-1.5 rounded uppercase">{d.camera}</span>
                  </div>
                  <div className="text-xs font-bold text-slate-200 uppercase">{d.type}</div>
                  <div className="text-[10px] text-slate-400">{d.location}</div>
                </div>
                
                <div className="flex-1 flex flex-col sm:items-end justify-center">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-[9px] text-slate-500 uppercase tracking-widest">Confidence</span>
                    <div className="w-16 h-1.5 bg-slate-800 rounded overflow-hidden">
                      <div className="h-full bg-cyan-500" style={{ width: `${d.confidence}%` }}></div>
                    </div>
                    <span className="text-[10px] font-mono text-slate-300">{d.confidence}%</span>
                  </div>
                  <span className={`text-[9px] px-1.5 py-0.5 rounded border uppercase font-bold tracking-widest ${getSeverityStyle(d.severity)}`}>
                    {d.severity}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Correlations Column */}
        <div className="lg:col-span-4 flex flex-col gap-6 h-[450px]">
          {/* Incident Correlation */}
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex-1 flex flex-col">
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 shrink-0">CCTV ↔ Incident Correlation</h2>
            <div className="flex-grow flex flex-col justify-center">
              {activeIncidents.length > 0 ? (
                <div className="bg-slate-950 p-3 border border-slate-800 rounded text-center">
                  <div className="text-xs font-bold text-slate-300 mb-1">CCTV Detection</div>
                  <div className="text-[10px] text-orange-400 uppercase mb-3">Suspicious Gathering</div>
                  <div className="text-slate-600 mb-3">↓</div>
                  <div className="text-xs font-bold text-slate-300 mb-1">Police Incident</div>
                  <div className="text-[10px] text-red-400 uppercase mb-4">{activeIncidents[0].incident_type.replace(/_/g, ' ')}</div>
                  <button onClick={() => navigate(`/police/incidents?id=${activeIncidents[0].incident_id}`)} className="text-[9px] font-bold uppercase tracking-widest bg-blue-900/30 text-blue-400 hover:bg-blue-900/50 border border-blue-800/50 py-1.5 px-3 rounded transition-colors w-full">
                    View Incident &rarr;
                  </button>
                </div>
              ) : (
                <span className="text-xs text-slate-500 text-center uppercase tracking-widest">No Active Incidents for Correlation</span>
              )}
            </div>
          </div>

          {/* Cross-Camera Correlation */}
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex-1 flex flex-col">
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3 shrink-0">Cross-Camera Pattern</h2>
            <div className="bg-purple-950/20 border border-purple-900/30 p-3 rounded flex-grow flex flex-col">
              <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 mb-2">
                <span>CAM-021 <span className="text-purple-400">18:41:32</span></span>
                <span>&rarr;</span>
                <span>CAM-112 <span className="text-purple-400">18:42:17</span></span>
              </div>
              <div className="text-[10px] font-bold text-slate-300 uppercase mt-auto mb-1">Correlated Event</div>
              <p className="text-[10px] text-slate-400">Movement detected across adjacent camera zones. Review sequence advised.</p>
            </div>
          </div>
        </div>

      </div>

      {/* HEALTH & QUEUE ROW */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mb-6">
        
        {/* Network Health & Analytics */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-4">
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">Camera Network Health</h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between text-[10px] uppercase font-bold text-slate-400 mb-1">
                <span>Network Integrity</span>
                <span className="font-mono text-emerald-400">94%</span>
              </div>
              <div className="w-full bg-slate-950 h-2 rounded flex overflow-hidden">
                <div className="bg-emerald-500 h-full" style={{ width: '94%' }}></div>
                <div className="bg-amber-400 h-full" style={{ width: '3%' }}></div>
                <div className="bg-red-500 h-full" style={{ width: '3%' }}></div>
              </div>
              <div className="flex justify-between text-[9px] font-mono text-slate-500">
                <span className="text-emerald-400">121 Online</span>
                <span className="text-amber-400">3 Degraded</span>
                <span className="text-red-400">4 Offline</span>
              </div>
            </div>
          </div>
          
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex-grow">
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">Detections by Type</h2>
            <div className="space-y-3">
              {Object.entries(analytics.typeDistribution).map(([type, count]) => {
                const pct = Math.round((count / 100) * 100);
                return (
                  <div key={type}>
                    <div className="flex justify-between text-[10px] text-slate-300 mb-1">
                      <span>{type}</span>
                      <span className="font-mono">{count}</span>
                    </div>
                    <div className="w-full bg-slate-950 h-1.5 rounded overflow-hidden">
                      <div className="bg-slate-500 h-full" style={{ width: `${pct}%` }}></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Operator Queue & Timeline */}
        <div className="lg:col-span-8 grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col h-[350px]">
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 shrink-0">Operator Queue</h2>
            <div className="flex-grow overflow-y-auto custom-scrollbar space-y-2 pr-2">
              {SYNTHETIC_QUEUE.map(task => (
                <div key={task.id} className="bg-slate-950 border border-slate-800 p-3 rounded flex flex-col gap-2">
                  <div className="flex justify-between items-center">
                    <span className={`text-[9px] px-1.5 py-0.5 rounded border uppercase font-bold tracking-widest ${task.priority === 'P1' ? 'text-red-400 border-red-500/30' : 'text-orange-400 border-orange-500/30'}`}>
                      {task.priority}
                    </span>
                    <span className="text-[10px] font-mono text-slate-500">{task.time}</span>
                  </div>
                  <div className="text-xs font-bold text-slate-200">{task.reason}</div>
                  <div className="flex justify-between items-center mt-1">
                    <span className="text-[10px] font-mono text-slate-400 bg-slate-900 px-1.5 rounded">{task.camera}</span>
                    <div className="flex gap-2">
                      <button className="text-[9px] text-slate-400 hover:text-white uppercase tracking-widest font-bold transition-colors">Acknowledge</button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col h-[350px]">
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 shrink-0">Agent Activity</h2>
            <div className="flex-grow overflow-y-auto custom-scrollbar pl-2 pr-2">
               <div className="border-l border-slate-700 ml-2 space-y-4 py-2">
                {SYNTHETIC_TIMELINE.map((evt, idx) => (
                  <div key={idx} className="relative pl-4">
                    <div className={`absolute -left-1.5 top-1.5 w-3 h-3 rounded-full border-2 border-slate-900 ${idx === 0 ? 'bg-cyan-500 animate-pulse' : 'bg-slate-600'}`}></div>
                    <div className="text-[10px] font-mono text-slate-500 mb-0.5">{evt.time}</div>
                    <div className={`text-[10px] font-bold uppercase tracking-widest ${idx === 0 ? 'text-cyan-400' : 'text-slate-300'}`}>{evt.event}</div>
                    <div className="text-xs text-slate-400 mt-0.5">{evt.desc}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

      </div>

      {/* CAMERA NETWORK TABLE */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden flex flex-col shadow-lg">
        <div className="p-4 border-b border-slate-800 flex justify-between items-center">
          <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest">Camera Network</h2>
          <input type="text" placeholder="Search cameras..." className="bg-slate-950 border border-slate-700 text-[10px] text-slate-200 px-3 py-1 rounded focus:outline-none focus:border-cyan-500 w-48 font-mono" />
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead className="bg-slate-950 text-[9px] uppercase tracking-widest text-slate-500 border-b border-slate-800">
              <tr>
                <th className="p-3 font-bold">Camera</th>
                <th className="p-3 font-bold">Location</th>
                <th className="p-3 font-bold">Status</th>
                <th className="p-3 font-bold">AI Monitoring</th>
                <th className="p-3 font-bold">Last Seen</th>
                <th className="p-3 font-bold">Alerts</th>
                <th className="p-3 font-bold">Risk</th>
                <th className="p-3 font-bold text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {SYNTHETIC_CAMERAS.map(cam => (
                <tr key={cam.id} className="hover:bg-slate-800/50 transition-colors">
                  <td className="p-3 text-xs font-mono font-bold text-slate-300">{cam.id}</td>
                  <td className="p-3 text-xs text-slate-400">{cam.location}</td>
                  <td className="p-3">
                    <span className={`text-[9px] font-bold uppercase tracking-widest ${cam.status === 'ONLINE' ? 'text-emerald-400' : cam.status === 'DEGRADED' ? 'text-amber-400' : 'text-red-400'}`}>
                      {cam.status}
                    </span>
                  </td>
                  <td className="p-3 text-[10px] font-mono text-cyan-400">{cam.aiMonitoring}</td>
                  <td className="p-3 text-[10px] font-mono text-slate-500">{cam.lastSeen}</td>
                  <td className="p-3 text-xs font-mono text-slate-300">{cam.alerts}</td>
                  <td className="p-3">
                     <span className={`text-[9px] font-bold uppercase tracking-widest ${getTextColor(cam.risk)}`}>{cam.risk}</span>
                  </td>
                  <td className="p-3 text-right">
                    <button className="text-[9px] font-bold uppercase tracking-widest text-slate-500 hover:text-cyan-400 transition-colors">Details &rarr;</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}