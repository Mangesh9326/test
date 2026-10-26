import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';

// Utility Icons
const BrainIcon = () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>;
const ActivityIcon = () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" /></svg>;
const AlertTriangle = () => <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>;

const ProgressBar = ({ occupied, total, colorClass }) => {
  if (total == null || total === 0) return <div className="h-1.5 w-full bg-slate-800 rounded-full"></div>;
  const pct = Math.min(((occupied || 0) / total) * 100, 100);
  return (
    <div className="flex flex-col gap-1 w-full">
      <div className="flex justify-between items-center text-[10px] text-slate-400 font-mono">
        <span>{occupied}/{total}</span>
        <span>{Math.round(pct)}%</span>
      </div>
      <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden border border-slate-700/50">
        <div className={`h-full transition-all duration-500 ${colorClass || 'bg-blue-500'}`} style={{ width: `${pct}%` }}></div>
      </div>
    </div>
  );
};

export default function HospitalOverview() {
  const navigate = useNavigate();
  const [data, setData] = useState({
    summary: null,
    monitoring: [],
    agentOverview: null,
    activeDecisions: [],
    history: []
  });
  
  const [loading, setLoading] = useState(true);
  const [isPaused, setIsPaused] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [countdown, setCountdown] = useState(30);

  const fetchNetworkIntelligence = useCallback(async () => {
    try {
      const responses = await Promise.allSettled([
        fetch('/api/hospitals/network-summary'),
        fetch('/api/hospitals/monitoring'),
        fetch('/api/hospital/agent/overview'),
        fetch('/api/hospital/agent/active'),
        fetch('/api/hospital/agent/decisions?limit=15')
      ]);

      const parsed = await Promise.all(responses.map(async (res) => {
        if (res.status === 'fulfilled' && res.value.ok) return await res.value.json();
        return null;
      }));

      setData({
        summary: parsed[0] || null,
        monitoring: parsed[1] || [],
        agentOverview: parsed[2] || null,
        activeDecisions: parsed[3] || [],
        history: parsed[4]?.data || []
      });
      
      setLastUpdated(new Date());
      setCountdown(30);
    } catch (err) {
      console.error('[HOSPITAL UI] Intelligence fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Auto-Refresh & Countdown Logic
  useEffect(() => {
    fetchNetworkIntelligence();
  }, [fetchNetworkIntelligence]);

  useEffect(() => {
    if (isPaused) return;
    const interval = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          fetchNetworkIntelligence();
          return 30;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, [isPaused, fetchNetworkIntelligence]);

  // Derived Metrics & Analysis (Performant via useMemo)
  const networkCondition = data.agentOverview?.networkCondition || 'NORMAL';
  const conditionColor = networkCondition === 'CRITICAL' ? 'text-red-500' : networkCondition === 'HIGH' ? 'text-orange-500' : networkCondition === 'MODERATE' ? 'text-yellow-400' : 'text-emerald-400';

  const emergencyStats = useMemo(() => {
    let totalCap = 0, totalOcc = 0, ready = 0, restricted = 0, unavailable = 0;
    data.monitoring.forEach(m => {
      totalCap += (m.emergency_capacity || 0);
      totalOcc += (m.emergency_occupancy || 0);
      if (!m.emergency_available) unavailable++;
      else if (m.emergency_level === 'CRITICAL' || m.emergency_level === 'HIGH') restricted++;
      else ready++;
    });
    return { totalCap, totalOcc, ready, restricted, unavailable };
  }, [data.monitoring]);

  const patientFlowStats = useMemo(() => {
    let avgWaitTotal = 0, waitCount = 0;
    const inflows = { NORMAL: 0, INCREASING: 0, HIGH: 0, CRITICAL: 0 };
    data.monitoring.forEach(m => {
      if (m.average_wait_minutes != null) { avgWaitTotal += m.average_wait_minutes; waitCount++; }
      if (inflows[m.patient_inflow] !== undefined) inflows[m.patient_inflow]++;
    });
    return { avgWait: waitCount ? Math.round(avgWaitTotal / waitCount) : 0, inflows };
  }, [data.monitoring]);

  const bottlenecks = useMemo(() => {
    const issues = [];
    const icuRatio = data.summary && data.summary.total_icu > 0 ? (data.summary.total_icu - data.summary.available_icu) / data.summary.total_icu : 0;
    if (icuRatio > 0.85) issues.push({ name: 'ICU Capacity', desc: `${data.monitoring.filter(m => m.icu_available <= 2).length} facilities severely restricted` });
    if (emergencyStats.totalCap > 0 && (emergencyStats.totalOcc / emergencyStats.totalCap) > 0.85) issues.push({ name: 'Emergency Load', desc: 'Sustained elevated ER occupancy across network' });
    if (patientFlowStats.avgWait > 25) issues.push({ name: 'Patient Inflow', desc: `Network-wide average wait time climbing (${patientFlowStats.avgWait}m)` });
    return issues;
  }, [data.summary, data.monitoring, emergencyStats, patientFlowStats]);

  const topDecision = data.activeDecisions.length > 0 ? data.activeDecisions[0] : null;
  const topMon = topDecision ? data.monitoring.find(m => m.hospital_id === topDecision.hospital_id) : null;

  if (loading && !data.summary) {
    return (
      <div className="flex h-full items-center justify-center bg-slate-950 font-mono text-sm text-slate-400">
        <span className="animate-pulse">Initializing Command Center...</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6 h-full p-4 sm:p-6 text-slate-200 bg-slate-950 overflow-y-auto custom-scrollbar font-sans">
      
      {/* SECTION 1 & 19 — OVERVIEW HEADER & AUTO REFRESH */}
      <header className="flex flex-col xl:flex-row justify-between items-start xl:items-center gap-4 shrink-0 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-3xl font-bold uppercase tracking-tight text-white">Mumbai Healthcare Network</h1>
          <p className="text-sm text-slate-400 mt-1 uppercase tracking-widest font-bold">Healthcare Operations & AI Intelligence</p>
          <div className="flex flex-wrap gap-2 mt-3 text-[10px] font-bold uppercase tracking-widest">
            <span className="px-2 py-1 bg-emerald-900/30 text-emerald-400 border border-emerald-800/50 rounded flex items-center gap-1.5"><span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse"></span> NETWORK ONLINE</span>
            <span className="px-2 py-1 bg-blue-900/30 text-blue-400 border border-blue-800/50 rounded flex items-center gap-1.5"><span className="w-1.5 h-1.5 bg-blue-500 rounded-full"></span> HOSPITAL AGENT ACTIVE</span>
            <span className="px-2 py-1 bg-slate-800 text-slate-400 border border-slate-700 rounded">DEMONSTRATION DATA</span>
          </div>
        </div>
        
        <div className="flex flex-col items-end gap-2 bg-slate-900/80 p-3 rounded-xl border border-slate-800">
          <div className="flex items-center gap-4 text-xs font-mono text-slate-400">
            <span>Last Assessment: {lastUpdated ? lastUpdated.toLocaleTimeString() : 'N/A'}</span>
            <span className={isPaused ? 'text-orange-400' : 'text-blue-400'}>{isPaused ? 'PAUSED' : `Next update in ${countdown}s`}</span>
          </div>
          <div className="flex gap-2 mt-1">
            <button onClick={() => setIsPaused(!isPaused)} className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-[10px] font-bold uppercase tracking-widest rounded border border-slate-700 transition-colors">
              {isPaused ? 'Resume' : 'Pause'}
            </button>
            <button onClick={fetchNetworkIntelligence} className="px-3 py-1.5 bg-blue-600/20 hover:bg-blue-600/40 text-blue-400 text-[10px] font-bold uppercase tracking-widest rounded border border-blue-500/30 transition-colors">
              Refresh Now
            </button>
          </div>
        </div>
      </header>

      {/* SECTION 2 — NETWORK KPI STRIP */}
      <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-7 gap-3 shrink-0">
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl shadow-sm flex flex-col justify-center">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Monitored</span>
          <span className="text-2xl font-mono font-bold text-slate-200 mt-1">{data.agentOverview?.monitoredHospitals || 0}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl shadow-sm flex flex-col justify-center">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Operational</span>
          <span className="text-2xl font-mono font-bold text-emerald-400 mt-1">{data.summary?.operational_hospitals || 0}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl shadow-sm flex flex-col justify-center">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Emergency Ready</span>
          <span className="text-2xl font-mono font-bold text-blue-400 mt-1">{data.summary?.emergency_ready_hospitals || 0}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl shadow-sm flex flex-col justify-center">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">High Pressure</span>
          <span className="text-2xl font-mono font-bold text-orange-400 mt-1">{data.summary?.high_pressure_hospitals || 0}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl shadow-sm flex flex-col justify-center">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Critical</span>
          <span className="text-2xl font-mono font-bold text-red-500 mt-1">{data.summary?.critical_hospitals || 0}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl shadow-sm flex flex-col justify-center">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Available Beds</span>
          <span className="text-2xl font-mono font-bold text-emerald-300 mt-1">{data.summary?.available_beds || 0}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-3.5 rounded-xl shadow-sm flex flex-col justify-center">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Available ICU</span>
          <span className="text-2xl font-mono font-bold text-blue-300 mt-1">{data.summary?.available_icu || 0}</span>
        </div>
      </div>

      {/* MAIN COMMAND CENTER GRID */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 flex-1 items-start">
        
        {/* COLUMN 1 */}
        <div className="flex flex-col gap-6">
          
          {/* SECTION 3 — NETWORK CONDITION */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">Network Condition</h3>
            <div className="flex items-center justify-between mb-2">
              <span className={`text-2xl font-bold uppercase ${conditionColor}`}>{networkCondition}</span>
            </div>
            <p className="text-xs text-slate-300">
              {data.summary?.high_pressure_hospitals > 0 
                ? `${data.summary.high_pressure_hospitals} hospitals showing elevated operational pressure across the network.` 
                : 'All monitored facilities operating within standard capacity thresholds.'}
            </p>
          </div>

          {/* SECTION 4 & 17 — HOSPITAL AGENT STATUS & PIPELINE */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm flex flex-col gap-5">
            <div>
              <div className="flex items-center gap-2 mb-4 text-blue-400">
                <BrainIcon />
                <h3 className="font-bold text-sm uppercase tracking-widest">Hospital Agent</h3>
              </div>
              <div className="grid grid-cols-2 gap-3 text-xs font-mono mb-4">
                <div className="bg-slate-950 p-2 rounded border border-slate-800"><span className="text-[9px] text-slate-500 uppercase tracking-widest block mb-1">Status</span><span className="text-emerald-400 font-bold">{data.agentOverview?.status || 'N/A'}</span></div>
                <div className="bg-slate-950 p-2 rounded border border-slate-800"><span className="text-[9px] text-slate-500 uppercase tracking-widest block mb-1">Mode</span><span className="text-blue-400 font-bold">{data.agentOverview?.mode || 'N/A'}</span></div>
                <div className="bg-slate-950 p-2 rounded border border-slate-800"><span className="text-[9px] text-slate-500 uppercase tracking-widest block mb-1">Assessments</span><span className="text-slate-200">{data.agentOverview?.activeAssessments || 0}</span></div>
                <div className="bg-slate-950 p-2 rounded border border-slate-800"><span className="text-[9px] text-slate-500 uppercase tracking-widest block mb-1">Attention Req</span><span className="text-orange-400">{data.agentOverview?.attentionRequired || 0}</span></div>
              </div>
            </div>

            <div className="pt-4 border-t border-slate-800">
               <h4 className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-3">Hospital Agent Pipeline</h4>
               <div className="flex justify-between items-center text-[8px] sm:text-[9px] font-mono font-bold text-slate-400 uppercase tracking-widest relative">
                 <div className="absolute top-1/2 left-0 w-full h-[1px] bg-slate-800 -z-10"></div>
                 {['Monitor', 'Analyze', 'Predict', 'Decide', 'Recommend'].map((step, i) => (
                    <div key={step} className="bg-slate-950 px-1.5 py-1 rounded border border-blue-900/50 text-blue-400 z-10">{step}</div>
                 ))}
               </div>
            </div>
          </div>

          {/* SECTION 11 — NETWORK CAPACITY */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2"><ActivityIcon /> Network Capacity</h3>
            <div className="flex flex-col gap-4">
              <ProgressBar label="General Beds" occupied={(data.summary?.total_beds || 0) - (data.summary?.available_beds || 0)} total={data.summary?.total_beds} colorClass="bg-emerald-500" />
              <ProgressBar label="Intensive Care (ICU)" occupied={(data.summary?.total_icu || 0) - (data.summary?.available_icu || 0)} total={data.summary?.total_icu} colorClass="bg-blue-500" />
              <ProgressBar label="Emergency Capacity" occupied={emergencyStats.totalOcc} total={emergencyStats.totalCap} colorClass="bg-orange-500" />
            </div>
          </div>

          {/* SECTION 12 — EMERGENCY NETWORK */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
             <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">Emergency Network Status</h3>
             <div className="grid grid-cols-3 gap-2 text-center text-xs font-mono">
               <div className="bg-slate-950 p-2 rounded border border-slate-800 flex flex-col justify-center">
                 <span className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">Ready</span>
                 <span className="text-emerald-400 font-bold">{emergencyStats.ready}</span>
               </div>
               <div className="bg-slate-950 p-2 rounded border border-slate-800 flex flex-col justify-center">
                 <span className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">Restricted</span>
                 <span className="text-orange-400 font-bold">{emergencyStats.restricted}</span>
               </div>
               <div className="bg-slate-950 p-2 rounded border border-slate-800 flex flex-col justify-center">
                 <span className="text-[10px] text-slate-500 uppercase tracking-widest mb-1">Unavailable</span>
                 <span className="text-slate-500 font-bold">{emergencyStats.unavailable}</span>
               </div>
             </div>
          </div>

        </div>

        {/* COLUMN 2 */}
        <div className="flex flex-col gap-6">
          
          {/* SECTION 6, 7, 8, 16 — EXPLAINABILITY & CURRENT ASSESSMENT */}
          <div className="bg-blue-950/10 border border-blue-900/40 rounded-xl shadow-sm overflow-hidden">
            <div className="bg-blue-900/20 border-b border-blue-900/40 p-4 flex justify-between items-center">
              <h3 className="text-xs font-bold text-blue-400 uppercase tracking-widest">Current Agent Assessment</h3>
              <span className="text-[9px] bg-blue-500/20 text-blue-300 px-2 py-0.5 rounded border border-blue-500/30 font-mono tracking-widest">EXPLAINABILITY</span>
            </div>
            
            {topDecision ? (
              <div className="p-5 flex flex-col gap-5">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="text-lg font-bold text-white uppercase">{topDecision.hospital_name}</h4>
                    <div className="flex gap-2 mt-1">
                      <span className={`text-[9px] font-bold uppercase tracking-widest px-2 py-0.5 rounded border ${topMon?.capacity_level === 'CRITICAL' ? 'bg-red-900/30 text-red-400 border-red-800/50' : 'bg-orange-900/30 text-orange-400 border-orange-800/50'}`}>Capacity: {topMon?.capacity_level || 'N/A'}</span>
                      <span className={`text-[9px] font-bold uppercase tracking-widest px-2 py-0.5 rounded border ${topMon?.emergency_level === 'CRITICAL' ? 'bg-red-900/30 text-red-400 border-red-800/50' : 'bg-orange-900/30 text-orange-400 border-orange-800/50'}`}>Emergency: {topMon?.emergency_level || 'N/A'}</span>
                    </div>
                  </div>
                  <button onClick={() => navigate('/hospital/hospitals')} className="p-2 bg-slate-800 hover:bg-slate-700 rounded text-slate-400 transition-colors" title="View in Directory">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" /></svg>
                  </button>
                </div>

                {/* Input Signals */}
                <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                  <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold block mb-2">Input Signals</span>
                  <div className="grid grid-cols-3 gap-2 text-xs font-mono">
                    <div><span className="text-slate-500">ER Occ:</span> <span className="text-slate-200">{topMon ? `${topMon.emergency_occupancy}/${topMon.emergency_capacity}` : 'N/A'}</span></div>
                    <div><span className="text-slate-500">ICU Occ:</span> <span className="text-slate-200">{topMon ? `${topMon.icu_occupied}/${topMon.icu_total}` : 'N/A'}</span></div>
                    <div><span className="text-slate-500">Avail Beds:</span> <span className="text-emerald-400">{topMon?.available_beds ?? 'N/A'}</span></div>
                    <div><span className="text-slate-500">Inflow:</span> <span className="text-slate-200">{topMon?.patient_inflow || 'N/A'}</span></div>
                    <div><span className="text-slate-500">Queue:</span> <span className="text-orange-400">{topMon?.ambulance_queue || 0}</span></div>
                    <div><span className="text-slate-500">Wait:</span> <span className="text-slate-200">{topMon?.average_wait_minutes ? `${topMon.average_wait_minutes}m` : 'N/A'}</span></div>
                  </div>
                </div>

                <div className="flex flex-col gap-3">
                  <div><span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold block mb-0.5">Assessment</span><p className="text-xs text-slate-300">{topDecision.situation}</p></div>
                  
                  {topDecision.prediction && (
                    <div className="bg-blue-900/10 p-2.5 rounded border border-blue-900/30">
                      <span className="text-[9px] text-blue-400 uppercase tracking-widest font-bold flex items-center gap-1.5 mb-1"><BrainIcon /> Prediction</span>
                      <p className="text-xs text-slate-300">Emergency load projected to reach <span className={topDecision.prediction.level === 'CRITICAL' ? 'text-red-400 font-bold' : 'text-orange-400 font-bold'}>{topDecision.prediction.level}</span> within {topDecision.prediction.horizon_minutes} minutes.</p>
                    </div>
                  )}

                  <div><span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold block mb-0.5">Decision: <span className="text-blue-400 ml-1">{topDecision.decision_type}</span></span><p className="text-xs text-emerald-400 font-medium">{topDecision.decision}</p></div>
                  <div><span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold block mb-0.5">Expected Impact</span><p className="text-xs text-slate-400 italic">{topDecision.expected_impact}</p></div>
                  
                  <div className="pt-3 border-t border-slate-800 flex justify-between items-center">
                    <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Confidence</span>
                    <span className="text-sm font-mono font-bold text-emerald-400">{topDecision.confidence}%</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-8 text-center text-slate-500 text-xs italic font-mono">No active critical assessments requiring explainability analysis.</div>
            )}
          </div>

          {/* SECTION 5 — AGENT ATTENTION QUEUE */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl flex flex-col shadow-sm overflow-hidden flex-1 max-h-[300px]">
            <div className="p-4 border-b border-slate-800 bg-slate-800/30 flex justify-between items-center shrink-0">
              <h3 className="font-bold text-xs uppercase tracking-widest text-slate-200 flex items-center gap-2"><AlertTriangle /> Agent Attention Queue</h3>
            </div>
            <div className="flex-1 overflow-y-auto custom-scrollbar p-3">
              {data.activeDecisions.length === 0 ? (
                 <div className="text-slate-500 text-xs italic text-center py-6">Queue is clear. No immediate attention required.</div>
              ) : (
                <div className="flex flex-col gap-2">
                  {data.activeDecisions.map(dec => (
                    <div key={dec.id} onClick={() => navigate('/hospital/hospitals')} className="bg-slate-950 border border-slate-800 p-3 rounded-lg hover:border-slate-600 cursor-pointer transition-colors">
                      <div className="flex justify-between items-start mb-1">
                        <span className={`text-[9px] font-bold uppercase tracking-widest px-2 py-0.5 rounded ${dec.status === 'ACTIVE' ? 'bg-orange-900/30 text-orange-400' : 'bg-slate-800 text-slate-400'}`}>{dec.decision_type}</span>
                        <span className="text-xs font-bold text-slate-200 uppercase">{dec.hospital_name}</span>
                      </div>
                      <p className="text-[10px] text-slate-400 mt-1.5 leading-snug truncate">{dec.situation}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

        </div>

        {/* COLUMN 3 */}
        <div className="flex flex-col gap-6">
          
          {/* SECTION 15 — AGENT RECOMMENDATIONS */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">Agent Recommendations</h3>
            {data.activeDecisions.length === 0 ? (
               <div className="text-slate-500 text-xs italic">No active recommendations.</div>
            ) : (
              <ul className="space-y-3">
                {data.activeDecisions.slice(0, 3).map((dec, i) => (
                  <li key={i} className="flex gap-2 text-xs text-slate-300">
                    <span className="text-blue-400 shrink-0">▸</span>
                    <span><strong className="text-slate-200">{dec.hospital_name}:</strong> {dec.recommendation}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* SECTION 14 — NETWORK BOTTLENECKS */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">Network Bottlenecks</h3>
            {bottlenecks.length === 0 ? (
               <div className="text-slate-500 text-xs italic">No systemic bottlenecks detected.</div>
            ) : (
              <div className="flex flex-col gap-3">
                {bottlenecks.map((b, i) => (
                  <div key={i} className="bg-red-950/20 border-l-2 border-red-500 p-2.5 rounded-r">
                    <span className="text-[10px] font-bold uppercase tracking-widest text-red-400 block mb-0.5">{b.name}</span>
                    <span className="text-xs text-slate-300">{b.desc}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* SECTION 13 — PATIENT FLOW */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
             <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">Patient Flow</h3>
             <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-800/50">
                <span className="text-xs text-slate-400">Network Average Wait:</span>
                <span className="text-lg font-mono font-bold text-orange-400">{patientFlowStats.avgWait}m</span>
             </div>
             <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
               <div className="bg-slate-950 p-2 rounded border border-slate-800 flex justify-between"><span className="text-slate-500">Normal</span><span className="text-emerald-400">{patientFlowStats.inflows.NORMAL}</span></div>
               <div className="bg-slate-950 p-2 rounded border border-slate-800 flex justify-between"><span className="text-slate-500">Increasing</span><span className="text-yellow-400">{patientFlowStats.inflows.INCREASING}</span></div>
               <div className="bg-slate-950 p-2 rounded border border-slate-800 flex justify-between"><span className="text-slate-500">High</span><span className="text-orange-400">{patientFlowStats.inflows.HIGH}</span></div>
               <div className="bg-slate-950 p-2 rounded border border-slate-800 flex justify-between"><span className="text-slate-500">Critical</span><span className="text-red-400">{patientFlowStats.inflows.CRITICAL}</span></div>
             </div>
          </div>

          {/* SECTION 18 — CROSS-AGENT READINESS */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">Cross-Agent Coordination</h3>
            <div className="flex flex-col gap-2 text-xs">
              <div className="flex justify-between items-center bg-slate-950 p-2 rounded border border-slate-800"><span className="text-slate-300">Hospital Agent</span><span className="text-[9px] font-mono text-emerald-400 bg-emerald-900/30 px-1.5 py-0.5 rounded border border-emerald-800/50">READY</span></div>
              <div className="flex justify-between items-center bg-slate-950 p-2 rounded border border-slate-800"><span className="text-slate-500">Traffic Agent</span><span className="text-[9px] font-mono text-slate-500 bg-slate-800 px-1.5 py-0.5 rounded border border-slate-700">READY FOR COORDINATION</span></div>
              <div className="flex justify-between items-center bg-slate-950 p-2 rounded border border-slate-800"><span className="text-slate-500">Police Agent</span><span className="text-[9px] font-mono text-slate-500 bg-slate-800 px-1.5 py-0.5 rounded border border-slate-700">READY FOR COORDINATION</span></div>
            </div>
          </div>

          {/* SECTION 10 & 9 — AGENT ACTIVITY / DECISION HISTORY */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl flex flex-col shadow-sm overflow-hidden flex-1 max-h-[300px]">
            <div className="p-4 border-b border-slate-800 bg-slate-800/30 flex justify-between items-center shrink-0">
              <h3 className="font-bold text-xs uppercase tracking-widest text-slate-200 flex items-center gap-2"><ActivityIcon /> Agent Activity</h3>
            </div>
            <div className="flex-1 overflow-y-auto custom-scrollbar p-4 bg-black">
              {data.history.length === 0 ? (
                <div className="text-slate-600 text-xs italic font-mono text-center py-4">No recent activity.</div>
              ) : (
                <div className="space-y-4 border-l border-slate-800 ml-2 pl-3">
                  {data.history.map((hist, i) => (
                    <div key={hist.id || i} className="relative">
                      <div className="absolute -left-[17px] top-1.5 w-2 h-2 rounded-full bg-blue-500 ring-2 ring-black"></div>
                      <div className="text-[9px] text-slate-500 font-mono mb-0.5">{new Date(hist.created_at).toLocaleTimeString()}</div>
                      <div className="text-xs text-slate-300 font-bold uppercase">{hist.hospital_name}</div>
                      <div className="text-[10px] text-slate-400 font-mono mt-0.5">{hist.decision_type} • Conf: {hist.confidence}%</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}