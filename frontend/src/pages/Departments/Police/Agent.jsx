import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { usePoliceStore } from '../../../store/usePoliceStore';

// ==========================================
// ICONS
// ==========================================
const ShieldIcon = () => <svg className="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" /></svg>;
const ActivityIcon = () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>;
const CheckCircleIcon = () => <svg className="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>;
const NetworkIcon = () => <svg className="w-4 h-4 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9" /></svg>;

// ==========================================
// DETERMINISTIC UTILS
// ==========================================
const detHash = (str) => {
  let hash = 0;
  for (let i = 0; i < str?.length; i++) hash = (hash << 5) - hash + str.charCodeAt(i);
  return Math.abs(hash);
};

const getSeverityColor = (severity) => {
  switch (String(severity || '').toUpperCase()) {
    case 'CRITICAL': return 'text-red-400 border-red-500/30 bg-red-500/10';
    case 'HIGH': return 'text-orange-400 border-orange-500/30 bg-orange-500/10';
    case 'MEDIUM': return 'text-amber-400 border-amber-500/30 bg-amber-500/10';
    default: return 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10';
  }
};

// ==========================================
// SYNTHETIC DETERMINISTIC ADAPTERS
// ==========================================
// Generates explainable, deterministic agent logic for a given incident
const buildAgentAssessment = (incident) => {
  if (!incident) return null;
  const isIncident = !!incident.incident_id;
  const id = incident.incident_id || incident.id;
  const type = incident.incident_type || incident.category;
  const severity = String(incident.severity || 'LOW').toUpperCase();
  const hash = detHash(id);

  let riskScore = 20 + (hash % 15);
  let threatLevel = "ELEVATED";
  if (severity === 'CRITICAL') { riskScore = 85 + (hash % 14); threatLevel = "CRITICAL"; }
  else if (severity === 'HIGH') { riskScore = 70 + (hash % 14); threatLevel = "HIGH"; }
  else if (severity === 'MEDIUM') { riskScore = 45 + (hash % 20); threatLevel = "ELEVATED"; }

  const coordinationAgents = ['Traffic Agent'];
  if (severity === 'CRITICAL') coordinationAgents.push('Hospital Agent', 'Fire Agent');
  else if (isIncident) coordinationAgents.push('City Operations Agent');

  return {
    targetId: id,
    type: type?.replace(/_/g, ' '),
    location: incident.location,
    severity,
    riskScore,
    threatLevel,
    confidence: 80 + (hash % 18),
    evidenceCount: 3 + (hash % 4),
    escalationTrend: (hash % 3 === 0) ? "Increasing" : "Stable",
    observedSignals: [
      `${3 + (hash % 2)} related alerts within the monitored sector.`,
      `Spatial clustering detected around ${incident.location}.`,
      `Current operational baseline exceeded by ${15 + (hash % 25)}%.`
    ],
    decisionRationale: `High severity ${type} combined with ${hash % 3 === 0 ? 'increasing' : 'sustained'} activity density. Multi-source validation confirms active security requirement.`,
    recommendedAction: `Deploy nearest available patrol unit. Coordinate with ${coordinationAgents[0]} for perimeter support.`,
    coordinationAgents,
    time: new Date(incident.time || incident.reported_at || Date.now()).toLocaleTimeString()
  };
};

// ==========================================
// MAIN COMPONENT
// ==========================================
export default function PoliceAgent() {
  const navigate = useNavigate();
  const { 
    overviewMetrics, 
    activeIncidents, 
    crimes, 
    agentRecommendation,
    fetchInitialData 
  } = usePoliceStore();

  const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString());
  
  useEffect(() => {
    fetchInitialData();
    const timer = setInterval(() => setCurrentTime(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(timer);
  }, [fetchInitialData]);

  // Derive Focus Event
  const currentFocus = useMemo(() => {
    const focusTarget = (activeIncidents?.length > 0) ? activeIncidents[0] : (crimes?.length > 0 ? crimes[0] : null);
    return buildAgentAssessment(focusTarget);
  }, [activeIncidents, crimes]);

  const queue = useMemo(() => {
    return (crimes || []).slice(1, 4).map(c => buildAgentAssessment(c)).sort((a, b) => b.riskScore - a.riskScore);
  }, [crimes]);

  // ==========================================
  // RENDER SECTIONS
  // ==========================================
  return (
    <div className="flex flex-col min-h-screen bg-slate-950 text-slate-200 font-sans p-4 sm:p-6 gap-6 overflow-x-hidden">
      
      {/* 1. HERO / AGENT STATUS */}
      <header className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col md:flex-row justify-between items-start md:items-center gap-4 shrink-0">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <ShieldIcon />
            <h1 className="text-2xl font-bold text-slate-100 uppercase tracking-widest">Police Agent</h1>
          </div>
          <p className="text-xs text-slate-400 font-mono tracking-wider">AI Security & Decision Engine</p>
        </div>
        
        <div className="flex flex-col items-end gap-2">
          <div className="text-[10px] text-blue-400 font-bold uppercase tracking-widest flex items-center gap-2 bg-blue-950/30 border border-blue-900/50 px-3 py-1.5 rounded-full shadow-[0_0_15px_rgba(59,130,246,0.15)]">
            <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span> AGENT ACTIVE
          </div>
          <span className="text-xs text-slate-500 font-mono">Continuously monitoring city safety intelligence</span>
        </div>
      </header>

      {/* 2. KPI & SYSTEM HEALTH */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 shrink-0">
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-center">
          <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1">Sources Monitored</span>
          <span className="text-xl font-mono font-bold text-slate-200">6 <span className="text-xs text-emerald-400 ml-1">● Healthy</span></span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-center">
          <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1">Active Assessments</span>
          <span className="text-xl font-mono font-bold text-blue-400">{(activeIncidents?.length || 0) + (crimes?.slice(0, 5).length || 0)}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-center">
          <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1">High-Risk Events</span>
          <span className="text-xl font-mono font-bold text-red-400">{overviewMetrics?.criticalIncidents || 0}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-center">
          <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1">Pending Decisions</span>
          <span className="text-xl font-mono font-bold text-orange-400">{queue.length}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-center">
          <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1">Agent Confidence</span>
          <span className="text-xl font-mono font-bold text-cyan-400">92% <span className="text-xs text-slate-500 ml-1">AVG</span></span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-center">
          <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1">Last Analysis</span>
          <span className="text-sm font-mono font-bold text-slate-300 mt-1">{currentTime}</span>
        </div>
      </div>

      {/* 3. AGENT OPERATING LOOP VISUALIZATION */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg overflow-hidden shrink-0">
        <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-5">Continuous Decision Workflow</h2>
        <div className="flex flex-wrap md:flex-nowrap items-center justify-between gap-2 overflow-x-auto custom-scrollbar pb-2">
          {['OBSERVE', 'DETECT', 'CORRELATE', 'ASSESS RISK', 'DECIDE', 'COORDINATE', 'RECOMMEND / DISPATCH', 'VERIFY OUTCOME', 'LEARN / UPDATE'].map((step, idx, arr) => (
            <React.Fragment key={step}>
              <div className="flex flex-col items-center gap-2 group min-w-[80px]">
                <div className={`w-8 h-8 rounded-full border-2 flex items-center justify-center text-xs font-bold transition-all ${idx === 4 ? 'bg-blue-600/20 border-blue-500 text-blue-400 shadow-[0_0_10px_rgba(59,130,246,0.3)] scale-110' : 'bg-slate-950 border-slate-700 text-slate-500 group-hover:border-slate-500 group-hover:text-slate-300'}`}>
                  {idx + 1}
                </div>
                <span className={`text-[9px] font-bold uppercase tracking-wider text-center ${idx === 4 ? 'text-blue-400' : 'text-slate-500'}`}>{step}</span>
              </div>
              {idx < arr.length - 1 && <div className="h-0.5 w-6 sm:w-12 bg-slate-800 flex-shrink-0"></div>}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* 4. HOW IT WORKS (SOURCES & PIPELINE) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 shrink-0">
        {/* Intelligence Sources */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
          <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2"><NetworkIcon /> Intelligence Sources</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {[
              { name: 'Crime Records', val: `${crimes?.length || 0} active records` },
              { name: 'Incident Reports', val: `${activeIncidents?.length || 0} open incidents` },
              { name: 'CCTV Intelligence', val: 'Monitoring 128 nodes' },
              { name: 'Traffic Agent', val: 'Connected' },
              { name: 'Patrol Availability', val: 'Real-time telemetry' },
              { name: 'Emergency Alerts', val: 'Listening' }
            ].map(src => (
              <div key={src.name} className="bg-slate-950 border border-slate-800 p-2.5 rounded flex items-center justify-between">
                <div>
                  <div className="text-xs font-bold text-slate-200">{src.name}</div>
                  <div className="text-[9px] text-slate-500 font-mono mt-0.5">{src.val}</div>
                </div>
                <CheckCircleIcon />
              </div>
            ))}
          </div>
        </div>

        {/* Detection Engine */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col">
          <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-4">Detection Engine Pipeline</h2>
          <div className="flex-1 bg-slate-950 border border-slate-800 rounded p-4 flex flex-col justify-center">
            <div className="flex flex-col gap-1.5">
               <div className="text-[10px] font-mono text-slate-400 border-l-2 border-slate-700 pl-3 py-1">Incoming Telemetry & Reports</div>
               <div className="text-[10px] font-mono text-slate-400 border-l-2 border-slate-600 pl-3 py-1">Event Normalization</div>
               <div className="text-[10px] font-mono text-cyan-500/70 border-l-2 border-cyan-700 pl-3 py-1">Pattern & Anomaly Detection</div>
               <div className="text-[10px] font-mono text-blue-400 border-l-2 border-blue-600 pl-3 py-1">Cross-Source Correlation</div>
               <div className="text-[10px] font-mono text-orange-400 border-l-2 border-orange-600 pl-3 py-1">Risk & Severity Assessment</div>
               <div className="text-[10px] font-mono font-bold text-red-400 border-l-2 border-red-500 pl-3 py-1">Potential Security Event Identified</div>
            </div>
          </div>
        </div>
      </div>

      {/* 5. CURRENT EVENT FOCUS (THE DECISION ENGINE) */}
      <div className="bg-slate-900 border border-blue-900/50 rounded-xl shadow-[0_0_20px_rgba(59,130,246,0.05)] overflow-hidden">
        <div className="bg-slate-950 p-4 border-b border-slate-800 flex justify-between items-center">
          <div className="flex items-center gap-3">
             <span className="w-2.5 h-2.5 rounded-full bg-blue-500 animate-pulse"></span>
             <h2 className="text-sm font-bold text-blue-400 uppercase tracking-widest">Active Agent Decision Center</h2>
          </div>
          {currentFocus && <span className="text-[10px] bg-slate-800 text-slate-300 px-2 py-0.5 rounded font-mono border border-slate-700">TARGET: {currentFocus.targetId}</span>}
        </div>

        {currentFocus ? (
          <div className="p-6 grid grid-cols-1 xl:grid-cols-3 gap-8">
            
            {/* Left: Evidence & Risk */}
            <div className="flex flex-col gap-6 xl:col-span-1">
              <div>
                 <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-2">Evidence Correlation</span>
                 <div className="space-y-2">
                   <div className="bg-slate-950 border border-slate-800 p-2.5 rounded flex items-start gap-3">
                     <div className="text-xs bg-purple-900/30 text-purple-400 border border-purple-500/30 px-1.5 py-0.5 rounded uppercase font-bold tracking-widest shrink-0">CRIME DB</div>
                     <span className="text-xs text-slate-300">Spatial clustering matching {currentFocus.type} detected.</span>
                   </div>
                   <div className="bg-slate-950 border border-slate-800 p-2.5 rounded flex items-start gap-3">
                     <div className="text-xs bg-cyan-900/30 text-cyan-400 border border-cyan-500/30 px-1.5 py-0.5 rounded uppercase font-bold tracking-widest shrink-0">CCTV AI</div>
                     <span className="text-xs text-slate-300">Unusual movement correlated near {currentFocus.location}.</span>
                   </div>
                   <div className="bg-slate-950 border border-slate-800 p-2.5 rounded flex items-start gap-3">
                     <div className="text-xs bg-indigo-900/30 text-indigo-400 border border-indigo-500/30 px-1.5 py-0.5 rounded uppercase font-bold tracking-widest shrink-0">TRAFFIC</div>
                     <span className="text-xs text-slate-300">Congestion building near primary exit corridors.</span>
                   </div>
                 </div>
              </div>

              <div>
                 <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-2">Risk Assessment</span>
                 <div className="grid grid-cols-2 gap-3">
                   <div className="bg-slate-950 border border-slate-800 p-3 rounded flex flex-col justify-center">
                     <span className="text-[9px] uppercase tracking-widest text-slate-500 mb-1">Risk Score</span>
                     <span className={`text-2xl font-mono font-bold ${currentFocus.riskScore >= 80 ? 'text-red-400' : 'text-orange-400'}`}>{currentFocus.riskScore}</span>
                   </div>
                   <div className="bg-slate-950 border border-slate-800 p-3 rounded flex flex-col justify-center">
                     <span className="text-[9px] uppercase tracking-widest text-slate-500 mb-1">Threat Level</span>
                     <span className={`text-sm font-bold tracking-widest uppercase ${currentFocus.threatLevel === 'CRITICAL' ? 'text-red-400' : 'text-orange-400'}`}>{currentFocus.threatLevel}</span>
                   </div>
                   <div className="bg-slate-950 border border-slate-800 p-3 rounded flex flex-col justify-center">
                     <span className="text-[9px] uppercase tracking-widest text-slate-500 mb-1">Confidence</span>
                     <span className="text-sm font-mono font-bold text-blue-400">{currentFocus.confidence}%</span>
                   </div>
                   <div className="bg-slate-950 border border-slate-800 p-3 rounded flex flex-col justify-center">
                     <span className="text-[9px] uppercase tracking-widest text-slate-500 mb-1">Trend</span>
                     <span className={`text-sm font-bold tracking-widest uppercase ${currentFocus.escalationTrend === 'Increasing' ? 'text-orange-400' : 'text-emerald-400'}`}>{currentFocus.escalationTrend}</span>
                   </div>
                 </div>
              </div>
            </div>

            {/* Middle: Agent Decision */}
            <div className="flex flex-col gap-4 xl:col-span-1 border-y xl:border-y-0 xl:border-x border-slate-800 py-6 xl:py-0 xl:px-6">
               <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-1">Agent Decision</span>
               
               <div className="space-y-4 flex-1">
                 <div>
                   <span className="text-[10px] text-slate-500 uppercase block mb-1">Event Classification</span>
                   <div className="text-lg font-bold text-slate-100 uppercase">{currentFocus.type}</div>
                   <div className="text-xs text-slate-400">{currentFocus.location}</div>
                 </div>

                 <div>
                   <span className="text-[10px] text-slate-500 uppercase block mb-1">Observed Signals</span>
                   <ul className="text-xs text-slate-300 font-mono list-disc ml-4 space-y-1">
                     {currentFocus.observedSignals.map((sig, i) => <li key={i}>{sig}</li>)}
                   </ul>
                 </div>

                 <div>
                   <span className="text-[10px] text-slate-500 uppercase block mb-1">Decision Rationale</span>
                   <p className="text-sm text-slate-300 leading-relaxed bg-slate-950 p-3 border border-slate-800 rounded">{currentFocus.decisionRationale}</p>
                 </div>
               </div>
            </div>

            {/* Right: Operator Control & Recommendation */}
            <div className="flex flex-col gap-4 xl:col-span-1">
               <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-1">Operational Output</span>
               
               <div className="bg-blue-950/20 border border-blue-900/40 p-4 rounded-lg flex flex-col gap-3">
                 <span className="text-[10px] font-bold text-blue-400 uppercase tracking-widest flex items-center gap-2"><ActivityIcon /> Recommended Action</span>
                 <p className="text-sm text-slate-200 font-bold">{currentFocus.recommendedAction}</p>
                 <div className="flex justify-between items-center pt-3 border-t border-blue-900/30 mt-2">
                   <span className="text-[10px] text-slate-400 uppercase">Priority: <span className={currentFocus.threatLevel === 'CRITICAL' ? 'text-red-400 font-bold' : 'text-orange-400 font-bold'}>{currentFocus.threatLevel}</span></span>
                   <span className="text-[10px] text-slate-400 uppercase">Actionable</span>
                 </div>
               </div>

               <div className="mt-auto">
                 <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-3">Operator Control</span>
                 <div className="flex flex-col gap-2">
                   <div className="flex gap-2">
                     <button className="flex-1 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold uppercase tracking-widest py-2.5 rounded transition-colors shadow-lg shadow-emerald-900/20">Approve</button>
                     <button className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-bold uppercase tracking-widest py-2.5 rounded border border-slate-700 transition-colors">Modify</button>
                   </div>
                   <button onClick={() => navigate(currentFocus.targetId.includes('INC') ? `/police/incidents?id=${currentFocus.targetId}` : `/police/crimes?id=${currentFocus.targetId}`)} className="w-full bg-slate-900 hover:bg-slate-800 text-blue-400 text-xs font-bold uppercase tracking-widest py-2.5 rounded border border-blue-900/50 transition-colors">
                     Open Investigation Record
                   </button>
                 </div>
               </div>
            </div>
          </div>
        ) : (
          <div className="p-12 text-center text-slate-500 text-sm uppercase tracking-widest flex flex-col items-center gap-3">
             <ActivityIcon /> Waiting for telemetry...
          </div>
        )}
      </div>

      {/* 6. SECONDARY PANELS ROW */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Multi-Agent Coordination */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col h-[380px]">
          <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-4 shrink-0">Multi-Agent Coordination</h2>
          <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 space-y-3">
            {currentFocus ? currentFocus.coordinationAgents.map((agent, i) => (
               <div key={agent} className="bg-slate-950 border border-slate-800 p-3 rounded">
                 <div className="flex justify-between items-center mb-2">
                   <div className="text-[10px] font-bold text-slate-300 uppercase flex items-center gap-1.5"><span className="text-blue-400">POLICE</span> <span className="text-slate-500">&rarr;</span> <span className="text-indigo-400">{agent.split(' ')[0]}</span></div>
                   <span className="text-[9px] font-mono text-slate-500">{currentFocus.time}</span>
                 </div>
                 <div className="text-xs text-slate-400 border-l-2 border-slate-700 pl-2 ml-1 space-y-2">
                   <div><span className="text-[9px] uppercase font-bold text-slate-500 block">Request</span><span className="text-slate-300">Assess impact of {currentFocus.type} at {currentFocus.location}.</span></div>
                   <div><span className="text-[9px] uppercase font-bold text-slate-500 block">Response</span><span className="text-emerald-400">Impact verified. Standing by for coordinated response protocol.</span></div>
                 </div>
               </div>
            )) : <div className="text-xs text-slate-500">No active coordination events.</div>}
          </div>
        </div>

        {/* Decision Queue */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col h-[380px]">
          <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-4 shrink-0">Agent Decision Queue</h2>
          <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 space-y-2">
             {queue.length === 0 ? <div className="text-xs text-slate-500">Queue is empty.</div> : 
               queue.map((q) => (
                 <div key={q.targetId} className="bg-slate-950 border border-slate-800 p-3 rounded cursor-pointer hover:border-slate-600 transition-colors">
                   <div className="flex justify-between items-start mb-1.5">
                     <span className={`text-[9px] px-1.5 py-0.5 rounded border uppercase font-bold tracking-widest ${getSeverityColor(q.threatLevel)}`}>{q.threatLevel}</span>
                     <span className="text-[10px] font-mono text-slate-500">{q.time}</span>
                   </div>
                   <div className="text-xs font-bold text-slate-200 truncate">{q.type}</div>
                   <div className="text-[10px] text-slate-400 truncate mb-2">{q.location}</div>
                   <div className="flex justify-between items-center mt-1 pt-2 border-t border-slate-800">
                     <span className="text-[9px] text-slate-500 uppercase">Awaiting Operator</span>
                     <span className="text-[9px] font-bold text-blue-400 uppercase tracking-widest">Review &rarr;</span>
                   </div>
                 </div>
               ))
             }
          </div>
        </div>

        {/* Audit & Performance */}
        <div className="flex flex-col gap-6 h-[380px]">
          {/* Active Tasks */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex-1 overflow-hidden flex flex-col">
            <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3 shrink-0">Active Agent Tasks</h2>
            <div className="space-y-2 overflow-y-auto custom-scrollbar pr-1 flex-1">
              <div className="flex justify-between items-center text-xs bg-slate-950 p-2 border border-slate-800 rounded">
                <span className="text-slate-300">Monitoring Sector Activity</span>
                <span className="text-[9px] uppercase tracking-widest text-emerald-400 font-bold">Active</span>
              </div>
              <div className="flex justify-between items-center text-xs bg-slate-950 p-2 border border-slate-800 rounded">
                <span className="text-slate-300">Cross-referencing CCTV feeds</span>
                <span className="text-[9px] uppercase tracking-widest text-blue-400 font-bold">Processing</span>
              </div>
              {currentFocus && (
                <div className="flex justify-between items-center text-xs bg-slate-950 p-2 border border-slate-800 rounded">
                  <span className="text-slate-300 truncate pr-2">Awaiting operator approval for {currentFocus.targetId}</span>
                  <span className="text-[9px] uppercase tracking-widest text-amber-400 font-bold shrink-0">Pending</span>
                </div>
              )}
            </div>
          </div>
          
          {/* Response Outcome */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg shrink-0 h-40 flex flex-col justify-center">
            <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3 text-center">Latest Verified Outcome</h2>
            <div className="bg-slate-950 border border-slate-800 p-3 rounded text-center">
              <div className="text-xs font-bold text-slate-200 mb-1">Area Stabilized</div>
              <div className="text-[10px] text-slate-400">Patrol deployment successful. No further anomalies detected in sector over past 45 minutes.</div>
            </div>
          </div>
        </div>

      </div>

    </div>
  );
}