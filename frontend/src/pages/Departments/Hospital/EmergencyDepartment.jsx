import { useState, useEffect, useMemo, useRef } from "react";
import { runSimulation, SCENARIOS } from "./components/EDSimulationEngine";
import { EDTable } from "./components/EDTable";
import { EDDetailPanel } from "./components/EDDetailPanel";

export default function EmergencyDepartment() {
  const [liveHospitals, setLiveHospitals] = useState([]);
  const [apiDecisions, setApiDecisions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [lastFetch, setLastFetch] = useState(null);
  
  const [mode, setMode] = useState("LIVE"); // LIVE or SIMULATION
  const [selectedId, setSelectedId] = useState(null);

  // Simulation State
  const [simConfig, setSimConfig] = useState(SCENARIOS[0]);
  const [simTime, setSimTime] = useState(0);
  const [simStatus, setSimStatus] = useState("READY");
  const [simSpeed, setSimSpeed] = useState(1);
  const simTimer = useRef(null);

  // 1. Live Data Management (Memory Leak & Double Fetch Fix via AbortController)
  useEffect(() => {
    let isMounted = true;
    let timeoutId;
    const controller = new AbortController();

    const fetchLive = async () => {
      try {
        const [hRes, dRes] = await Promise.all([
          fetch("/api/hospitals/emergency", { signal: controller.signal }),
          fetch("/api/hospitals/hospital/agent/decisions?limit=5", { signal: controller.signal })
        ]);
        if (!hRes.ok) throw new Error("API Error");
        
        const hData = await hRes.json();
        const dData = dRes.ok ? await dRes.json() : { data: [] };

        if (isMounted) {
          setLiveHospitals(hData);
          setApiDecisions(dData.data || []);
          setLastFetch(new Date());
          setError(false);
        }
      } catch (err) {
        if (isMounted && err.name !== 'AbortError') setError(true);
      } finally {
        if (isMounted) {
          setLoading(false);
          timeoutId = setTimeout(fetchLive, 30000);
        }
      }
    };

    fetchLive();
    return () => {
      isMounted = false;
      clearTimeout(timeoutId);
      controller.abort();
    };
  }, []);

  // 2. Simulation Loop Controller
  useEffect(() => {
    if (simStatus === "RUNNING") {
      simTimer.current = setInterval(() => {
        setSimTime(prev => {
          const next = prev + 1;
          if (next >= simConfig.duration) {
            setSimStatus("COMPLETED");
            return simConfig.duration;
          }
          return next;
        });
      }, 1000 / simSpeed);
    }
    return () => clearInterval(simTimer.current);
  }, [simStatus, simSpeed, simConfig.duration]);

  // 3. Compute active state cleanly
  const currentSimData = useMemo(() => {
    if (mode === "LIVE" || simStatus === "READY") return null;
    return runSimulation(liveHospitals, simConfig, simTime);
  }, [mode, simStatus, simTime, liveHospitals, simConfig]);

  const displayHospitals = mode === "LIVE" ? liveHospitals : (currentSimData?.hospitals || liveHospitals);
  const selectedHospital = displayHospitals.find(h => h.hospital_id === selectedId) || displayHospitals[0];

  // Derive Live KPIs deterministically
  const liveNetworkStatus = useMemo(() => {
    if (mode === "SIMULATION" && currentSimData) return currentSimData;
    
    let capacity = 0, occupancy = 0, queue = 0, waitTotal = 0, waitCount = 0;
    let criticalCount = 0;

    liveHospitals.forEach(h => {
      capacity += Number(h.emergency_capacity || 0);
      occupancy += Number(h.emergency_occupancy || 0);
      queue += Number(h.ambulance_queue || 0);
      if (h.average_wait_minutes != null) { waitTotal += Number(h.average_wait_minutes); waitCount++; }
      if (h.capacity_level === "CRITICAL") criticalCount++;
    });

    const netCond = criticalCount >= 2 ? "CRITICAL" : criticalCount === 1 ? "HIGH" : "NORMAL";

    return {
      capacity, occupancy, queue, 
      wait: waitCount ? Math.round(waitTotal / waitCount) : null,
      networkCondition: netCond,
      alerts: criticalCount > 0 ? [{ level: "CRITICAL", text: `${criticalCount} hospital(s) showing critical pressure.` }] : []
    };
  }, [mode, liveHospitals, currentSimData]);

  // Handlers
  const handleModeSwitch = (newMode) => {
    setMode(newMode);
    if (newMode === "SIMULATION") { setSimStatus("READY"); setSimTime(0); }
  };

  const handleSimControl = (cmd) => {
    if (cmd === 'START') setSimStatus("RUNNING");
    if (cmd === 'PAUSE') setSimStatus("PAUSED");
    if (cmd === 'RESET') { setSimStatus("READY"); setSimTime(0); }
  };

  const operationalStages = mode === "SIMULATION" && currentSimData
    ? Object.entries(currentSimData.flow).map(([label, count]) => ({ label, value: count, active: currentSimData.bottleneck.includes(label.toUpperCase()) }))
    : [
        { label: "Arrival", value: liveNetworkStatus.queue == null ? "N/A" : `${liveNetworkStatus.queue} queued`, active: liveNetworkStatus.queue > 0 },
        { label: "Registration", value: "Telemetry unavailable", active: false },
        { label: "Triage", value: "Telemetry unavailable", active: false },
        { label: "Waiting", value: liveNetworkStatus.wait == null ? "N/A" : `${liveNetworkStatus.wait} min`, active: liveNetworkStatus.wait >= 20 },
        { label: "Treatment", value: liveNetworkStatus.capacity ? `${liveNetworkStatus.occupancy} active` : "N/A", active: liveNetworkStatus.capacity && liveNetworkStatus.occupancy / liveNetworkStatus.capacity >= 0.75 },
        { label: "Observation", value: "Telemetry unavailable", active: false },
        { label: "Admission / Discharge", value: "Telemetry unavailable", active: false },
      ];

  if (loading && liveHospitals.length === 0) {
    return <div className="flex min-h-screen items-center justify-center font-mono text-slate-400 text-sm animate-pulse bg-slate-950">Synchronizing Emergency Network...</div>;
  }

  return (
    // FIX: Using min-h-screen and standard scrolling instead of overflow-hidden
    <div className="min-h-screen pb-12 p-4 sm:p-6 lg:p-8 bg-slate-950 text-slate-200 font-sans space-y-6">
      
      {/* HEADER */}
      <header className="flex flex-col lg:flex-row justify-between items-start lg:items-end gap-4 border-b border-slate-800/80 pb-6">
        <div>
          <h1 className="text-3xl font-bold uppercase tracking-tight text-white">Emergency Department</h1>
          <p className="text-sm text-slate-400 mt-2 uppercase tracking-widest font-bold">Network Operations • Patient Flow • Decision Support</p>
          <p className="text-xs text-slate-500 mt-2 max-w-2xl">Use this command view to identify capacity strain, choose a facility from the ranked network list, and review the Hospital Agent’s recommendation before coordinating response teams.</p>
        </div>
        <div className="flex flex-col items-end gap-3 w-full lg:w-auto">
          {error && <span className="text-[10px] font-bold text-red-400 bg-red-900/20 border border-red-800 px-3 py-1.5 rounded w-full lg:w-auto text-center">LIVE DATA UNAVAILABLE - SHOWING LAST SUCCESS</span>}
          <div className="flex bg-slate-900 border border-slate-700 rounded-lg overflow-hidden p-1 w-full lg:w-auto">
            <button onClick={() => handleModeSwitch("LIVE")} className={`flex-1 lg:flex-none px-6 py-2 text-xs font-bold tracking-widest uppercase rounded transition-colors focus:outline-none ${mode === "LIVE" ? "bg-emerald-600 text-white shadow" : "text-slate-400 hover:text-slate-200"}`}>LIVE OPERATIONS</button>
            <button onClick={() => handleModeSwitch("SIMULATION")} className={`flex-1 lg:flex-none px-6 py-2 text-xs font-bold tracking-widest uppercase rounded transition-colors focus:outline-none ${mode === "SIMULATION" ? "bg-indigo-600 text-white shadow" : "text-slate-400 hover:text-slate-200"}`}>SIMULATION</button>
          </div>
        </div>
      </header>

      {/* STATUS STRIP */}
      <div className={`border rounded-lg p-4 text-[10px] sm:text-xs font-bold tracking-widest uppercase flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 ${mode === "SIMULATION" ? "border-indigo-500/40 bg-indigo-500/10 text-indigo-300" : "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"}`}>
        <div className="flex items-center gap-2.5">
          <span className={`w-2 h-2 rounded-full animate-pulse ${mode === "SIMULATION" ? "bg-indigo-400 shadow-[0_0_6px_rgba(129,140,248,0.8)]" : "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]"}`}></span>
          {mode === "SIMULATION" ? "DEMONSTRATION MODE • SIMULATION DATA • NO REAL HOSPITAL CONTROL" : "LIVE OPERATIONS • DATABASE / API DATA"}
        </div>
        <span className="text-slate-400 font-mono">Last Update: {lastFetch ? lastFetch.toLocaleTimeString() : 'N/A'}</span>
      </div>

      <section className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3" aria-label="How the Emergency Department Operations Center works">
        {[["1. Monitor", "Review incoming queue, occupancy, wait time, and alerts."], ["2. Identify", "Select a hospital ranked by emergency pressure."], ["3. Assess", "Inspect capacity signals and the Hospital Agent assessment."], ["4. Coordinate", "Use recommendations as decision support; no external controls are issued."]].map(([step, text]) => <div key={step} className="bg-slate-900/70 border border-slate-800 rounded-lg p-3"><p className="text-[10px] font-bold uppercase tracking-widest text-blue-400">{step}</p><p className="mt-1.5 text-xs text-slate-400 leading-relaxed">{text}</p></div>)}
      </section>

      {/* SIMULATION CONTROLS */}
      {mode === "SIMULATION" && (
        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-xl grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8 gap-5 items-end shadow-md">
           <label className="flex flex-col gap-2 text-[10px] text-slate-500 font-bold uppercase tracking-widest col-span-2">
             Scenario
             <select disabled={simStatus === "RUNNING"} value={simConfig.id} onChange={(e) => {setSimConfig(SCENARIOS.find(s=>s.id === e.target.value)); setSimStatus("READY"); setSimTime(0);}} className="bg-slate-950 border border-slate-700 rounded-lg px-3 py-2.5 text-xs text-slate-200 outline-none focus:border-indigo-500">
               {SCENARIOS.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
             </select>
           </label>
           <div className="flex flex-col gap-2 text-[10px] text-slate-500 font-bold uppercase tracking-widest">
             <span className="mb-0.5">Time</span>
             <span className="font-mono text-sm sm:text-base text-indigo-400 bg-slate-950 px-3 py-2 rounded-lg border border-slate-800">T+{String(simTime).padStart(3, '0')} / {simConfig.duration}s</span>
           </div>
           <div className="flex flex-col gap-2 text-[10px] text-slate-500 font-bold uppercase tracking-widest">
             <span className="mb-0.5">Phase</span>
             <span className="font-mono text-sm sm:text-base text-slate-200 bg-slate-950 px-3 py-2 rounded-lg border border-slate-800 truncate">{currentSimData?.phase || "READY"}</span>
           </div>
           <label className="flex flex-col gap-2 text-[10px] text-slate-500 font-bold uppercase tracking-widest">
             Sim Speed
             <select value={simSpeed} onChange={(e) => setSimSpeed(Number(e.target.value))} className="bg-slate-950 border border-slate-700 rounded-lg px-3 py-2.5 text-xs text-slate-200 outline-none focus:border-indigo-500">
               {[0.5, 1, 2, 5].map(x => <option key={x} value={x}>{x}x</option>)}
             </select>
           </label>
           <div className="col-span-2 xl:col-span-3 flex gap-3 h-[40px]">
             {simStatus === "READY" || simStatus === "COMPLETED" ? (
               <button onClick={() => handleSimControl('START')} className="flex-1 bg-indigo-600 hover:bg-indigo-500 text-white text-[10px] font-bold uppercase tracking-widest rounded-lg shadow-md transition-transform active:scale-95">START SCENARIO</button>
             ) : simStatus === "RUNNING" ? (
               <button onClick={() => handleSimControl('PAUSE')} className="flex-1 bg-orange-600/20 border border-orange-500/50 text-orange-400 hover:bg-orange-600/30 text-[10px] font-bold uppercase tracking-widest rounded-lg transition-colors">PAUSE</button>
             ) : (
               <button onClick={() => handleSimControl('START')} className="flex-1 bg-emerald-600/20 border border-emerald-500/50 text-emerald-400 hover:bg-emerald-600/30 text-[10px] font-bold uppercase tracking-widest rounded-lg transition-colors">RESUME</button>
             )}
             <button onClick={() => handleSimControl('RESET')} disabled={simStatus==="READY"} className="px-6 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 border border-slate-700 text-slate-300 text-[10px] font-bold uppercase tracking-widest rounded-lg transition-colors">RESET</button>
           </div>
        </div>
      )}

      {/* KPI ROW */}
      <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-4">
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-center shadow-sm">
          <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Network Condition</span>
          <span className={`text-2xl font-bold uppercase mt-1.5 ${liveNetworkStatus.networkCondition === 'CRITICAL' ? 'text-red-500' : liveNetworkStatus.networkCondition === 'HIGH' ? 'text-orange-400' : 'text-emerald-400'}`}>{liveNetworkStatus.networkCondition}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-center shadow-sm">
          <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Total ER Occupancy</span>
          <span className="text-2xl font-mono text-slate-200 mt-1.5">{liveNetworkStatus.occupancy} / {liveNetworkStatus.capacity || 'N/A'}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-center shadow-sm">
          <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Avg Wait Time</span>
          <span className="text-2xl font-mono text-orange-400 mt-1.5">{liveNetworkStatus.wait != null ? `${liveNetworkStatus.wait}m` : 'N/A'}</span>
        </div>
        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-center shadow-sm">
          <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Inbound Queue</span>
          <span className="text-2xl font-mono text-red-400 mt-1.5">{liveNetworkStatus.queue}</span>
        </div>
        {mode === "SIMULATION" && currentSimData && (
          <>
            <div className="bg-indigo-950/20 border border-indigo-900/40 p-4 rounded-xl flex flex-col justify-center shadow-sm">
              <span className="text-[10px] text-indigo-400/80 uppercase tracking-widest font-bold">Simulated Critical Load</span>
              <span className="text-2xl font-mono text-red-500 mt-1.5">{currentSimData.classification.critical}</span>
            </div>
            <div className="bg-indigo-950/20 border border-indigo-900/40 p-4 rounded-xl flex flex-col justify-center shadow-sm">
              <span className="text-[10px] text-indigo-400/80 uppercase tracking-widest font-bold">Current Bottleneck</span>
              <span className="text-sm font-bold text-orange-400 mt-2.5 truncate uppercase">{currentSimData.bottleneck}</span>
            </div>
          </>
        )}
      </div>

      {/* MAIN GRID LAYOUT - FULLY SCROLLABLE */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
        
        {/* LEFT COLUMN: Data Tables & Flow (Spans 2 columns on lg) */}
        <div className="lg:col-span-2 space-y-6">
          
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800/50 pb-3 mb-4"><div><h3 className="text-[10px] font-bold text-slate-300 uppercase tracking-widest">Emergency Patient Flow</h3><p className="mt-1 text-[11px] text-slate-500">{mode === "SIMULATION" ? "Deterministic simulation stage counts." : "Live stage counts are shown only where supplied by the API."}</p></div><span className="text-[10px] font-mono text-slate-500">{mode === "SIMULATION" ? currentSimData?.phase || "READY" : "LIVE TELEMETRY"}</span></div>
            <div className="flex overflow-x-auto gap-3 pb-2 custom-scrollbar">{operationalStages.map((stage, i) => <div key={stage.label} className="flex-1 min-w-[112px] bg-slate-950 border border-slate-800 rounded-lg p-3 text-center relative">{i > 0 && <div className="absolute top-1/2 -left-3 w-3 h-px bg-slate-700"/>}<span className={`mx-auto mb-2 block h-2 w-2 rounded-full ${stage.active ? "bg-orange-400 animate-pulse" : "bg-slate-700"}`}/><span className="text-[9px] text-slate-500 uppercase tracking-widest block">{stage.label}</span><span className="mt-1.5 block text-xs font-mono font-bold text-slate-200">{stage.value}</span></div>)}</div>
          </div>

          {/* Active Alerts */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
            <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest border-b border-slate-800/50 pb-3 mb-4">Active Emergency Alerts</h3>
            <div className="flex flex-col gap-3 text-sm">
              {liveNetworkStatus.alerts.length === 0 ? (
                <span className="text-slate-500 italic font-mono text-xs">No active network alerts.</span>
              ) : (
                liveNetworkStatus.alerts.map((a, i) => (
                  <div key={i} className="flex items-center gap-3 bg-slate-950/50 p-2 rounded">
                    <span className={`px-2 py-1 rounded border text-[10px] font-bold tracking-wider ${a.level === 'CRITICAL' ? 'bg-red-500/10 border-red-500/30 text-red-400' : 'bg-orange-500/10 border-orange-500/30 text-orange-400'}`}>{a.level}</span>
                    <span className="text-slate-300 font-medium">{a.text}</span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Hospital Pressure Table */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
            <div className="p-4 border-b border-slate-800 bg-slate-900 flex justify-between items-center">
              <h3 className="font-bold text-xs uppercase tracking-widest text-slate-200">Hospitals Ranked By ER Pressure</h3>
            </div>
            <EDTable hospitals={displayHospitals} selectedId={selectedId} onSelect={setSelectedId} isSimulation={mode==="SIMULATION"} />
          </div>

        </div>

        {/* RIGHT COLUMN: Detail Panel & Trace (Sticky) */}
        <div className="lg:col-span-1 space-y-6 sticky top-6">
          
          {/* Detail Panel */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm">
            <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-5">Hospital Emergency Profile</h3>
            <EDDetailPanel hospital={selectedHospital} isSimulation={mode==="SIMULATION"} />
          </div>

          {/* AI Decision Support Trace */}
          <div className="bg-blue-950/20 border border-blue-900/30 rounded-xl p-5 shadow-sm flex flex-col gap-4">
             <h3 className="text-[10px] font-bold text-blue-400 uppercase tracking-widest flex items-center gap-2 mb-1">
               <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse shadow-[0_0_6px_rgba(59,130,246,0.6)]"></span>
               AI DECISION SUPPORT TRACE
             </h3>

             {mode === "LIVE" ? (
               apiDecisions.length > 0 ? (
                 <div className="flex flex-col gap-4 text-xs">
                   <div className="text-[9px] text-slate-500 uppercase tracking-widest border-b border-slate-800 pb-2">Latest Database Assessment</div>
                   <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800">
                     <span className="text-blue-400 font-bold uppercase tracking-wider">{apiDecisions[0].decision_type}</span>
                     <p className="text-slate-300 mt-2 leading-relaxed">{apiDecisions[0].recommendation}</p>
                     <div className="flex justify-between items-center text-[10px] font-mono text-slate-500 mt-3 border-t border-slate-800/80 pt-2">
                       <span>CONF: {apiDecisions[0].confidence}%</span>
                       <span>{new Date(apiDecisions[0].created_at).toLocaleTimeString()}</span>
                     </div>
                   </div>
                 </div>
               ) : <span className="text-xs text-slate-500 italic">No recent decisions in database.</span>
             ) : (
               currentSimData?.agent ? (
                 <div className="flex flex-col text-xs font-mono gap-1.5">
                   <div className="bg-slate-950 p-2.5 rounded border border-slate-800 mb-1"><span className="text-[9px] text-slate-500 block mb-1 uppercase tracking-widest font-bold">Input Signals</span><span className="text-slate-300 leading-snug">{currentSimData.agent.signals}</span></div>
                   <div className="text-center text-slate-600 text-[10px]">↓</div>
                   <div className="bg-slate-950 p-2.5 rounded border border-slate-800 mb-1"><span className="text-[9px] text-slate-500 block mb-1 uppercase tracking-widest font-bold">Assessment</span><span className="text-red-400 font-bold">{currentSimData.agent.condition}</span></div>
                   <div className="text-center text-slate-600 text-[10px]">↓</div>
                   <div className="bg-slate-950 p-2.5 rounded border border-slate-800 mb-1"><span className="text-[9px] text-slate-500 block mb-1 uppercase tracking-widest font-bold">Prediction</span><span className="text-orange-400">{currentSimData.agent.prediction}</span></div>
                   <div className="text-center text-slate-600 text-[10px]">↓</div>
                   <div className="bg-blue-900/20 p-2.5 rounded border border-blue-900/50 mb-1"><span className="text-[9px] text-blue-400 block mb-1 uppercase tracking-widest font-bold">Recommendation</span><span className="text-slate-200 font-sans leading-relaxed">{currentSimData.agent.recommendation}</span></div>
                   <div className="flex justify-between items-center text-[10px] text-slate-500 mt-2 px-1"><span>CONF: {currentSimData.agent.confidence}%</span><span>{currentSimData.agent.timestamp}</span></div>
                 </div>
               ) : <span className="text-xs text-slate-500 italic font-mono p-4 bg-slate-900/50 rounded border border-slate-800">Awaiting capacity pressure thresholds...</span>
             )}
          </div>

          {/* Timeline (Simulation Only) */}
          {mode === "SIMULATION" && currentSimData && (
             <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm flex flex-col">
               <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest border-b border-slate-800/50 pb-3 mb-4">Simulation Timeline</h3>
               <div className="flex-1 relative pl-1">
                 <div className="absolute left-[5px] top-2 bottom-2 w-px bg-slate-800 -z-10"></div>
                 {currentSimData.timeline.map((item, i) => (
                   <div key={i} className="relative pl-6 pb-5 last:pb-0">
                     <div className={`absolute left-0 top-1 w-2.5 h-2.5 rounded-full border-2 border-slate-900 ${i === 0 ? 'bg-indigo-400 animate-pulse ring-2 ring-indigo-500/50' : 'bg-slate-600'}`}></div>
                     <div className="flex justify-between items-center mb-1">
                       <span className={`text-[10px] font-bold uppercase tracking-widest ${i===0 ? 'text-indigo-400' : 'text-slate-500'}`}>{item.event}</span>
                       <span className="text-[9px] font-mono text-slate-500 bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800">T+{Math.round(simConfig.duration * item.timePct)}s</span>
                     </div>
                     <p className={`text-xs leading-relaxed ${i===0 ? 'text-slate-200 font-medium' : 'text-slate-500'}`}>{item.effect}</p>
                   </div>
                 ))}
               </div>
             </div>
          )}

        </div>
      </div>
    </div>
  );
}
