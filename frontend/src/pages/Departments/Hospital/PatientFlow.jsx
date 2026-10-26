import React, { useState, useEffect, useMemo } from 'react';

export default function PatientFlow() {
  const [networkData, setNetworkData] = useState([]);
  const [selectedHospital, setSelectedHospital] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  // Filters
  const [filterHospital, setFilterHospital] = useState('All');

  useEffect(() => {
    const fetchPatientFlow = async () => {
      try {
        setLoading(true);
        setError(false);
        
        // Fetch base flow indicators and full monitoring context for queue/wait metrics
        const [flowRes, monitoringRes] = await Promise.all([
          fetch('/api/hospitals/patient-flow'),
          fetch('/api/hospitals/monitoring')
        ]);

        if (!flowRes.ok || !monitoringRes.ok) throw new Error('API Error');

        const flowData = await flowRes.json();
        const monitoringData = await monitoringRes.json();

        // Merge backend data to provide a unified flow view without inventing data
        const merged = monitoringData.map(mon => {
          const flow = flowData.find(f => f.name === mon.name) || {};
          return { ...mon, ...flow };
        }).sort((a, b) => a.name.localeCompare(b.name));

        setNetworkData(merged);
        if (merged.length > 0 && !selectedHospital) {
          setSelectedHospital(merged[0]);
        }
      } catch (err) {
        console.error('[HOSPITAL UI] Failed to load patient flow:', err);
        setError(true);
      } finally {
        setLoading(false);
      }
    };

    fetchPatientFlow();
    const interval = setInterval(fetchPatientFlow, 30000);
    return () => clearInterval(interval);
  }, [selectedHospital]);

  // Dynamic Filters
  const uniqueHospitals = useMemo(() => networkData.map(h => h.name), [networkData]);
  
  // Bottleneck Detection Logic (based strictly on available API fields)
  const bottleneck = useMemo(() => {
    if (networkData.length === 0) return null;
    const sortedByWait = [...networkData].sort((a, b) => (b.average_wait_minutes || 0) - (a.average_wait_minutes || 0));
    const worst = sortedByWait[0];
    
    if (worst && (worst.average_wait_minutes > 15 || worst.ambulance_queue > 2)) {
      return {
        hospital: worst.name,
        stage: worst.ambulance_queue > 0 ? 'Arrival / Ambulance Receiving' : 'Triage',
        queue: worst.ambulance_queue,
        wait: worst.average_wait_minutes,
        trend: worst.predicted_inflow || 'Unknown'
      };
    }
    return null;
  }, [networkData]);

  // Update selected hospital when filter changes
  useEffect(() => {
    if (filterHospital !== 'All') {
      const target = networkData.find(h => h.name === filterHospital);
      if (target) setSelectedHospital(target);
    }
  }, [filterHospital, networkData]);

  if (loading) return <div className="flex h-full items-center justify-center font-mono text-slate-400 animate-pulse">Scanning Network Patient Flow...</div>;
  if (error) return <div className="p-6 text-red-400 font-bold uppercase tracking-widest text-sm">Error connecting to Hospital API</div>;

  // Pipeline Definition mapping to selected hospital data
  const pipelineStages = selectedHospital ? [
    { id: 'arrival', label: 'Arrival', queue: selectedHospital.ambulance_queue, wait: 'N/A', count: selectedHospital.patient_inflow, time: 'N/A', status: selectedHospital.ambulance_queue > 0 ? 'WARNING' : 'NORMAL' },
    { id: 'registration', label: 'Registration', queue: 'N/A', wait: 'N/A', count: 'N/A', time: 'N/A', status: 'UNAVAILABLE' },
    { id: 'triage', label: 'Triage', queue: 'N/A', wait: `${selectedHospital.average_wait_minutes} min`, count: 'N/A', time: 'N/A', status: selectedHospital.average_wait_minutes > 15 ? 'WARNING' : 'NORMAL' },
    { id: 'doctor', label: 'Doctor Assessment', queue: 'N/A', wait: 'N/A', count: 'N/A', time: 'N/A', status: 'UNAVAILABLE' },
    { id: 'diagnostics', label: 'Diagnostics', queue: 'N/A', wait: 'N/A', count: 'N/A', time: 'N/A', status: 'UNAVAILABLE' },
    { id: 'treatment', label: 'Treatment', queue: 'N/A', wait: 'N/A', count: selectedHospital.emergency_occupancy, time: 'N/A', status: selectedHospital.capacity_level === 'CRITICAL' ? 'CRITICAL' : 'NORMAL' },
    { id: 'admission', label: 'Admission', queue: 'N/A', wait: 'N/A', count: 'N/A', time: 'N/A', status: 'UNAVAILABLE' },
    { id: 'discharge', label: 'Discharge', queue: 'N/A', wait: 'N/A', count: 'N/A', time: 'N/A', status: 'UNAVAILABLE' }
  ] : [];

  return (
    <div className="flex flex-col gap-6 h-full p-6 text-slate-200 bg-slate-950 overflow-y-auto font-sans">
      
      <header className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4 shrink-0 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-3xl font-bold uppercase tracking-tight text-white">Patient Flow</h1>
          <p className="text-sm text-slate-400 mt-1">Pipeline Tracking & Bottleneck Detection</p>
        </div>
        <div className="flex items-center gap-4">
          <select 
            value={filterHospital} 
            onChange={(e) => setFilterHospital(e.target.value)} 
            className="bg-slate-900 border border-slate-700 text-sm font-bold rounded px-4 py-2 outline-none text-slate-200"
          >
            <option value="All" disabled>Select Hospital...</option>
            {uniqueHospitals.map(h => <option key={h} value={h}>{h}</option>)}
          </select>
        </div>
      </header>

      {/* Bottleneck Alert */}
      {bottleneck ? (
        <div className="bg-red-950/40 border-l-4 border-red-500 p-5 rounded-r-xl shadow-lg shrink-0 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></span>
              <span className="text-red-400 font-bold uppercase tracking-widest text-xs">BOTTLENECK DETECTED</span>
            </div>
            <h2 className="text-xl font-bold text-slate-100 uppercase">{bottleneck.hospital}</h2>
            <span className="text-sm text-slate-400">Stage: <span className="text-slate-200 font-bold uppercase">{bottleneck.stage}</span></span>
          </div>
          <div className="flex gap-6 border-t md:border-t-0 md:border-l border-red-900/50 pt-4 md:pt-0 md:pl-6">
            <div className="flex flex-col"><span className="text-[10px] text-red-300 uppercase tracking-widest font-bold">Queue</span><span className="text-xl font-mono text-white">{bottleneck.queue}</span></div>
            <div className="flex flex-col"><span className="text-[10px] text-red-300 uppercase tracking-widest font-bold">Avg Wait</span><span className="text-xl font-mono text-white">{bottleneck.wait}m</span></div>
            <div className="flex flex-col"><span className="text-[10px] text-red-300 uppercase tracking-widest font-bold">Trend</span><span className="text-xl font-mono text-white uppercase">{bottleneck.trend}</span></div>
          </div>
        </div>
      ) : (
        <div className="bg-emerald-950/20 border border-emerald-900/50 p-4 rounded-xl flex items-center gap-3 shrink-0">
           <span className="w-2 h-2 bg-emerald-500 rounded-full"></span>
           <span className="text-emerald-400 font-bold uppercase tracking-widest text-xs">NETWORK FLOW STABLE. NO SEVERE BOTTLENECKS DETECTED.</span>
        </div>
      )}

      {/* Selected Hospital Flow Pipeline */}
      {selectedHospital && (
        <div className="flex flex-col flex-1 bg-slate-900 border border-slate-800 rounded-xl shadow-lg overflow-hidden">
          <div className="p-5 border-b border-slate-800 bg-slate-800/30 flex justify-between items-center shrink-0">
            <div>
              <h3 className="font-bold text-lg uppercase text-white">{selectedHospital.name}</h3>
              <span className="text-xs text-slate-400 uppercase tracking-widest">Live Flow Pipeline</span>
            </div>
            <span className="text-[10px] bg-slate-950 px-2 py-1 rounded text-slate-500 font-mono border border-slate-800">
              TIME-SERIES DATA: PARTIAL
            </span>
          </div>

          <div className="p-6 overflow-y-auto custom-scrollbar flex-1 bg-black">
            <div className="relative border-l-2 border-slate-800 ml-4 md:ml-6 space-y-8 pb-4">
              
              {pipelineStages.map((stage, idx) => (
                <div key={stage.id} className="relative pl-6 md:pl-10">
                  {/* Status Node */}
                  <div className={`absolute -left-[9px] top-1.5 w-4 h-4 rounded-full border-4 border-black ${
                    stage.status === 'CRITICAL' ? 'bg-red-500' :
                    stage.status === 'WARNING' ? 'bg-orange-500' :
                    stage.status === 'UNAVAILABLE' ? 'bg-slate-700' : 'bg-emerald-500'
                  }`}></div>
                  
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-950 p-4 rounded-lg border border-slate-800 shadow">
                    
                    <div className="w-48 shrink-0">
                      <span className="text-xs text-slate-500 font-bold uppercase tracking-widest block mb-1">Stage {idx + 1}</span>
                      <h4 className={`text-sm font-bold uppercase ${stage.status === 'UNAVAILABLE' ? 'text-slate-500' : 'text-slate-200'}`}>
                        {stage.label}
                      </h4>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 flex-1">
                      <div className="flex flex-col">
                        <span className="text-[9px] text-slate-500 uppercase font-bold tracking-widest mb-1">Count / Inflow</span>
                        <span className={`text-sm font-mono font-bold ${stage.count === 'N/A' ? 'text-slate-600' : 'text-blue-400 uppercase'}`}>{stage.count}</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-[9px] text-slate-500 uppercase font-bold tracking-widest mb-1">Queue</span>
                        <span className={`text-sm font-mono font-bold ${stage.queue === 'N/A' ? 'text-slate-600' : 'text-orange-400'}`}>{stage.queue}</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-[9px] text-slate-500 uppercase font-bold tracking-widest mb-1">Avg Wait</span>
                        <span className={`text-sm font-mono font-bold ${stage.wait === 'N/A' ? 'text-slate-600' : 'text-red-400'}`}>{stage.wait}</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-[9px] text-slate-500 uppercase font-bold tracking-widest mb-1">Processing Time</span>
                        <span className={`text-sm font-mono font-bold ${stage.time === 'N/A' ? 'text-slate-600' : 'text-emerald-400'}`}>{stage.time}</span>
                      </div>
                    </div>
                    
                    {stage.status === 'UNAVAILABLE' && (
                      <div className="hidden md:flex shrink-0 items-center justify-center">
                        <span className="text-[9px] text-slate-600 font-mono border border-slate-800 px-2 py-0.5 rounded uppercase">Sensor Offline</span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              
            </div>
          </div>
        </div>
      )}
    </div>
  );
}