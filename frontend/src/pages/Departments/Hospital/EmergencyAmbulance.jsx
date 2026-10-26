import React, { useState, useEffect, useMemo, useRef } from 'react';

// ==========================================
// DETERMINISTIC SCENARIO DEFINITIONS
// ==========================================
const PREDEFINED_SCENARIOS = [
  { id: 'S1', name: 'Major Road Accident', type: 'TRAUMA', severity: 'HIGH', patients: 3, critical: 1, ambs: 2, duration: 480, dest: 'Sion Hospital' },
  { id: 'S2', name: 'Multi-Vehicle Collision', type: 'TRAUMA', severity: 'CRITICAL', patients: 5, critical: 2, ambs: 3, duration: 600, dest: 'KEM Hospital' },
  { id: 'S3', name: 'Cardiac Emergency', type: 'MEDICAL', severity: 'CRITICAL', patients: 1, critical: 1, ambs: 1, duration: 400, dest: 'Lilavati Hospital' },
  { id: 'S4', name: 'Building Collapse', type: 'DISASTER', severity: 'CRITICAL', patients: 12, critical: 4, ambs: 5, duration: 900, dest: 'JJ Hospital' },
  { id: 'S5', name: 'Industrial Accident', type: 'HAZMAT', severity: 'HIGH', patients: 4, critical: 2, ambs: 3, duration: 540, dest: 'KEM Hospital' },
  { id: 'S6', name: 'Fire Casualty', type: 'FIRE', severity: 'HIGH', patients: 6, critical: 2, ambs: 3, duration: 600, dest: 'Sion Hospital' },
  { id: 'S7', name: 'Flood Emergency', type: 'ENVIRONMENTAL', severity: 'MODERATE', patients: 8, critical: 1, ambs: 4, duration: 720, dest: 'Cooper Hospital' },
  { id: 'S8', name: 'Mass Casualty Incident', type: 'DISASTER', severity: 'CRITICAL', patients: 20, critical: 8, ambs: 10, duration: 1200, dest: 'KEM Hospital' }
];

export default function EmergencyAmbulance() {
  // Base Data State
  const [apiAmbulances, setApiAmbulances] = useState([]);
  const [apiIncidents, setApiIncidents] = useState([]);
  const [selectedAmbulance, setSelectedAmbulance] = useState(null);
  const [selectedIncident, setSelectedIncident] = useState(null);
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);

  // Filter State
  const [incidentSearch, setIncidentSearch] = useState('');
  const [incidentPriority, setIncidentPriority] = useState('All');
  const [incidentStatus, setIncidentStatus] = useState('All');

  // Simulation State Engine
  const [isSimMode, setIsSimMode] = useState(false);
  const [simStatus, setSimStatus] = useState('READY'); // READY, RUNNING, PAUSED, COMPLETE
  const [simTime, setSimTime] = useState(0);
  const [simSpeed, setSimSpeed] = useState(1);
  const [selectedScenarioId, setSelectedScenarioId] = useState('S1');
  const [simOverrides, setSimOverrides] = useState({ severity: '', patients: '', ambs: '' });
  const [simAgentIntel, setSimAgentIntel] = useState(null);
  const simTimerRef = useRef(null);

  // Fetch Database (Base) Data
  useEffect(() => {
    let isMounted = true;
    const fetchData = async () => {
      try {
        const [ambRes, incRes] = await Promise.all([
          fetch('http://localhost:5000/api/hospitals/ambulances'),
          fetch('http://localhost:5000/api/hospitals/incidents')
        ]);
        if (!ambRes.ok || !incRes.ok) throw new Error(` API failed: ${ ambRes.status || incRes.status }`);
        const ambData = await ambRes.json();
        const incData = await incRes.json();
        
        if (isMounted) {
          setApiAmbulances(ambData);
          setApiIncidents(incData);
          setLastUpdated(new Date());
          setError(false);
        }
      } catch (err) {
        if (isMounted) setError(true);
      } finally {
        if (isMounted) setLoading(false);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 20000);
    return () => { isMounted = false; clearInterval(interval); };
  }, []);

  // Compute Active Simulation Entities (Purely Deterministic)
  const activeSimConfig = useMemo(() => {
    const base = PREDEFINED_SCENARIOS.find(s => s.id === selectedScenarioId) || PREDEFINED_SCENARIOS[0];
    return {
      ...base,
      severity: simOverrides.severity || base.severity,
      patients: simOverrides.patients ? parseInt(simOverrides.patients, 10) : base.patients,
      ambs: simOverrides.ambs ? parseInt(simOverrides.ambs, 10) : base.ambs,
    };
  }, [selectedScenarioId, simOverrides]);

  // Phase 7: Fetch Backend Agent Intelligence upon Sim Start
  useEffect(() => {
    if (simStatus === 'RUNNING' && simTime === 1 && isSimMode) {
       fetch('/api/hospitals/agent/evaluate-emergency', {
         method: 'POST',
         headers: { 'Content-Type': 'application/json' },
         body: JSON.stringify({
           incident_type: activeSimConfig.type,
           location: 'Simulated Zone',
           severity: activeSimConfig.severity,
           destination_hospital: activeSimConfig.dest
         })
       })
       .then(r => r.json())
       .then(data => setSimAgentIntel(data))
       .catch(err => console.error('[AGENT API] Evaluation failed:', err));
    }
  }, [simStatus, simTime, isSimMode, activeSimConfig]);

  const activeSimData = useMemo(() => {
    if (!isSimMode || simStatus === 'READY') return { incidents: [], ambulances: [], logs: [], currentState: 'READY' };
    
    const duration = activeSimConfig.duration;
    const pct = Math.min(simTime / duration, 1);
    
    // Deterministic Stage Mapping
    let state = 'REPORTED', incStatus = 'REPORTED';
    if (pct >= 1) state = 'AMBULANCE_AVAILABLE';
    else if (pct >= 0.85) state = 'RETURNING';
    else if (pct >= 0.80) state = 'HANDOVER_COMPLETED';
    else if (pct >= 0.75) state = 'HANDOVER_STARTED';
    else if (pct >= 0.70) state = 'HOSPITAL_ARRIVAL';
    else if (pct >= 0.60) state = 'HOSPITAL_APPROACH';
    else if (pct >= 0.35) state = 'TRANSPORT_STARTED';
    else if (pct >= 0.30) state = 'PATIENTS_ONBOARD';
    else if (pct >= 0.20) state = 'ARRIVED_AT_SCENE';
    else if (pct >= 0.10) state = 'AMBULANCE_EN_ROUTE';
    else if (pct >= 0.05) state = 'AMBULANCE_DISPATCHED';
    else if (pct >= 0.02) state = 'INCIDENT_VALIDATED';
    else state = 'INCIDENT_REPORTED';

    if (pct >= 0.75) incStatus = 'CLOSED';
    else if (pct >= 0.20) incStatus = 'AT_SCENE';
    else if (pct >= 0.05) incStatus = 'DISPATCHED';
    else if (pct >= 0.02) incStatus = 'VALIDATING';

    const simInc = {
      incident_id: 'INC-SIM-001',
      incident_type: activeSimConfig.name,
      location: 'Simulated Zone (Mumbai)',
      priority: activeSimConfig.severity,
      incident_status: incStatus,
      patient_count: activeSimConfig.patients,
      critical_patient_count: activeSimConfig.critical,
      ambulances_required: activeSimConfig.ambs,
      ambulances_assigned: pct >= 0.05 ? activeSimConfig.ambs : 0,
      receiving_hospital_name: activeSimConfig.dest,
      reported_at: new Date().toISOString(),
      isSimulated: true
    };

    const simAmbs = Array.from({ length: activeSimConfig.ambs }).map((_, i) => ({
      ambulance_id: `AMB-SIM-${i+1}`,
      incident_id: (pct >= 0.05 && pct < 0.85) ? 'INC-SIM-001' : null,
      priority: activeSimConfig.severity,
      status: state,
      recommended_hospital: (pct >= 0.05 && pct < 0.85) ? activeSimConfig.dest : null,
      eta_minutes: (state === 'AMBULANCE_EN_ROUTE' || state === 'TRANSPORT_STARTED' || state === 'HOSPITAL_APPROACH') 
        ? Math.max(0, Math.floor((duration * (state === 'AMBULANCE_EN_ROUTE' ? 0.20 : 0.70) - simTime) / 60)) 
        : null,
      isSimulated: true
    }));

    const milestones = [
      { t: 0, s: 'INCIDENT_REPORTED', m: `Incident reported: ${activeSimConfig.name}` },
      { t: 0.02, s: 'INCIDENT_VALIDATED', m: 'Incident validated and prioritized.' },
      { t: 0.05, s: 'AMBULANCE_DISPATCHED', m: `${activeSimConfig.ambs} ambulance(s) dispatched to scene.` },
      { t: 0.10, s: 'AMBULANCE_EN_ROUTE', m: 'Ambulance(s) en route. Telemetry active.' },
      { t: 0.20, s: 'ARRIVED_AT_SCENE', m: 'Ambulance(s) arrived at scene. Triage started.' },
      { t: 0.30, s: 'PATIENTS_ONBOARD', m: 'Patients secured onboard. Vitals transmitting.' },
      { t: 0.35, s: 'TRANSPORT_STARTED', m: `Transporting to ${activeSimConfig.dest}.` },
      { t: 0.60, s: 'HOSPITAL_APPROACH', m: 'Approaching hospital. Trauma bay alerted.' },
      { t: 0.70, s: 'HOSPITAL_ARRIVAL', m: 'Arrived at receiving facility.' },
      { t: 0.75, s: 'HANDOVER_STARTED', m: 'Patient handover to ER team initiated.' },
      { t: 0.80, s: 'HANDOVER_COMPLETED', m: 'Patient handover completed. Documentation filed.' },
      { t: 0.85, s: 'RETURNING', m: 'Ambulance(s) returning to home station.' },
      { t: 1.0, s: 'AMBULANCE_AVAILABLE', m: 'Ambulance(s) marked available for dispatch.' }
    ];

    const generatedLogs = [];
    milestones.forEach(m => {
      if (pct >= m.t) {
        generatedLogs.unshift({ 
          time: Math.floor(duration * m.t), 
          state: m.s, 
          text: m.m,
          incidentId: simInc.incident_id,
          ambulanceId: activeSimConfig.ambs > 1 ? `MULTIPLE (${activeSimConfig.ambs})` : 'AMB-SIM-1',
          timestamp: `T+${Math.floor(duration * m.t)}s`,
          status: m.s === 'HANDOVER_COMPLETED' || m.s === 'AMBULANCE_AVAILABLE' ? 'COMPLETED' : 'ACTIVE'
        });
      }
    });

    return { incidents: [simInc], ambulances: simAmbs, logs: generatedLogs, currentState: state };
  }, [isSimMode, simStatus, simTime, activeSimConfig]);

  // Combine Real + Simulated Data
  const displayIncidents = useMemo(() => [...apiIncidents, ...activeSimData.incidents], [apiIncidents, activeSimData.incidents]);
  const displayAmbulances = useMemo(() => [...apiAmbulances, ...activeSimData.ambulances], [apiAmbulances, activeSimData.ambulances]);

  // Simulation Loop Controller
  useEffect(() => {
    if (simStatus === 'RUNNING') {
      simTimerRef.current = setInterval(() => {
        setSimTime(prev => {
          const next = prev + 1;
          if (next >= activeSimConfig.duration) {
            setSimStatus('COMPLETE');
            return activeSimConfig.duration;
          }
          return next;
        });
      }, 1000 / simSpeed);
    }
    return () => clearInterval(simTimerRef.current);
  }, [simStatus, simSpeed, activeSimConfig.duration]);

  // Operational KPIs
  const kpis = useMemo(() => {
    const uniqueIncidents = new Set();
    const criticalIncidents = new Set();
    const receivingHospitals = new Set();
    let enRoute = 0, atScene = 0, inTransit = 0, available = 0, attentionReq = 0;
    let totalEta = 0, etaCount = 0;

    displayAmbulances.forEach(a => {
      if (a.incident_id) {
        uniqueIncidents.add(a.incident_id);
        if (a.priority === 'CRITICAL') criticalIncidents.add(a.incident_id);
      }
      if (a.status === 'AMBULANCE_EN_ROUTE' || a.status === 'AMBULANCE_DISPATCHED' || a.status === 'EN_ROUTE' || a.status === 'DISPATCHED') enRoute++;
      else if (a.status === 'ARRIVED_AT_SCENE' || a.status === 'AT_SCENE') atScene++;
      else if (a.status === 'TRANSPORT_STARTED' || a.status === 'PATIENTS_ONBOARD' || a.status === 'HOSPITAL_APPROACH' || a.status === 'TRANSPORTING') inTransit++;
      else if (a.status === 'AMBULANCE_AVAILABLE' || a.status === 'AVAILABLE') available++;

      if (a.recommended_hospital && !a.status.includes('AVAILABLE')) receivingHospitals.add(a.recommended_hospital);

      if (a.eta_minutes != null && a.eta_minutes > 0) {
        totalEta += a.eta_minutes;
        etaCount++;
      }

      if (!a.status.includes('AVAILABLE')) {
        if (a.hospital_capacity_status === 'CRITICAL' || a.hospital_emergency_status === 'CRITICAL' || (a.priority === 'CRITICAL' && a.eta_minutes > 20)) {
          attentionReq++;
        }
      }
    });

    return {
      activeIncidents: displayIncidents.length || uniqueIncidents.size,
      criticalIncidents: displayIncidents.filter(i => i.priority === 'CRITICAL').length || criticalIncidents.size,
      enRoute, atScene, inTransit, available,
      avgEta: etaCount > 0 ? Math.round(totalEta / etaCount) : null,
      receivingHospitals: receivingHospitals.size,
      attentionReq
    };
  }, [displayAmbulances, displayIncidents]);

  const filteredIncidents = useMemo(() => {
    return displayIncidents.filter(inc => {
      if (incidentPriority !== 'All' && inc.priority !== incidentPriority.toUpperCase()) return false;
      if (incidentStatus !== 'All') {
        const s = inc.incident_status || '';
        const f = incidentStatus.toUpperCase().replace(' ', '_');
        if (f === 'ACTIVE' && s === 'CLOSED') return false;
        else if (f !== 'ACTIVE' && !s.includes(f)) return false;
      }
      if (incidentSearch) {
        const q = incidentSearch.toLowerCase();
        return (inc.incident_id || '').toLowerCase().includes(q) || (inc.location || '').toLowerCase().includes(q) || (inc.incident_type || '').toLowerCase().includes(q);
      }
      return true;
    });
  }, [displayIncidents, incidentSearch, incidentPriority, incidentStatus]);

  const toggleSimulationMode = () => {
    setIsSimMode(!isSimMode);
    if (isSimMode) {
      setSimStatus('READY');
      setSimTime(0);
      setSimAgentIntel(null);
      setSelectedIncident(null);
      setSelectedAmbulance(null);
    }
  };

  const handleSimAction = (action) => {
    if (action === 'START') setSimStatus('RUNNING');
    else if (action === 'PAUSE') setSimStatus('PAUSED');
    else if (action === 'RESUME') setSimStatus('RUNNING');
    else if (action === 'RESET') {
      setSimStatus('READY');
      setSimTime(0);
      setSimAgentIntel(null);
    }
  };

  // Timeline Component
  const TimelineEvent = ({ event, isNewest }) => (
    <div className={`relative pl-4 pb-4 border-l border-slate-700/50 last:border-0 ${isNewest ? 'animate-pulse' : ''}`}>
      <div className={`absolute -left-1.5 top-0 w-3 h-3 rounded-full border-2 border-slate-900 ${isNewest ? 'bg-blue-400 shadow-[0_0_8px_rgba(96,165,250,0.8)]' : 'bg-slate-600'}`}></div>
      <div className="flex flex-col gap-1 -mt-1.5">
        <div className="flex justify-between items-center">
          <span className={`text-[10px] font-bold tracking-widest uppercase ${isNewest ? 'text-blue-400' : 'text-slate-500'}`}>{event.state.replace(/_/g, ' ')}</span>
          <span className="text-[9px] text-slate-500 font-mono">{event.timestamp}</span>
        </div>
        <span className={`text-xs ${isNewest ? 'text-slate-200' : 'text-slate-400'}`}>{event.text}</span>
        {(event.incidentId || event.ambulanceId) && (
          <div className="flex gap-2 mt-1">
            {event.incidentId && <span className="text-[9px] bg-slate-800/50 text-slate-400 px-1.5 py-0.5 rounded border border-slate-700/50 font-mono">{event.incidentId}</span>}
            {event.ambulanceId && <span className="text-[9px] bg-slate-800/50 text-slate-400 px-1.5 py-0.5 rounded border border-slate-700/50 font-mono">{event.ambulanceId}</span>}
          </div>
        )}
      </div>
    </div>
  );

  if (loading && displayAmbulances.length === 0 && displayIncidents.length === 0) {
    return (
      <div className="flex h-full items-center justify-center bg-slate-950 text-slate-400 font-mono text-sm uppercase tracking-widest animate-pulse">
        Synchronizing Operations Center...
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5 h-full p-4 sm:p-6 text-slate-200 bg-slate-950 overflow-hidden font-sans relative">
      
      {/* HEADER */}
      <header className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4 shrink-0 border-b border-slate-800/80 pb-4">
        <div>
          <h1 className="text-2xl font-bold uppercase tracking-tight text-white">Emergency & Ambulance</h1>
          <p className="text-sm text-slate-400 mt-1 uppercase tracking-widest font-bold">Trauma Routing & Fleet Coordination</p>
          <div className="flex flex-wrap items-center gap-3 mt-2 text-[10px] font-bold uppercase tracking-widest text-slate-400">
            <span className="text-slate-500">Last Updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : '...'}</span>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <button 
            onClick={toggleSimulationMode}
            className={`px-4 py-2 text-[10px] font-bold uppercase tracking-widest rounded transition-colors shadow-sm focus:outline-none ${isSimMode ? 'bg-indigo-900/40 text-indigo-400 border border-indigo-500/50' : 'bg-slate-900 text-slate-300 border border-slate-700 hover:bg-slate-800'}`}
          >
            {isSimMode ? 'Exit Simulation Center' : 'Enter Simulation Mode'}
          </button>
          {isSimMode ? (
            <span className="px-3 py-1.5 bg-indigo-600 text-white text-[10px] font-bold uppercase tracking-widest rounded shadow-[0_0_10px_rgba(79,70,229,0.5)]">
              DEMONSTRATION / SIMULATION MODE
            </span>
          ) : (
            <span className="px-3 py-1.5 bg-slate-900 border border-slate-700 text-slate-400 text-[10px] font-bold uppercase tracking-widest rounded shadow-sm">
              DEMONSTRATION DATA
            </span>
          )}
          <span className="px-3 py-1.5 bg-blue-900/20 border border-blue-800/50 text-blue-400 text-[10px] font-bold uppercase tracking-widest rounded flex items-center gap-2 shadow-sm">
            <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse shadow-[0_0_6px_rgba(59,130,246,0.6)]"></span> HOSPITAL AGENT ACTIVE
          </span>
        </div>
      </header>

      {/* OPERATIONAL KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-5 xl:grid-cols-10 gap-3 shrink-0">
        <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg flex flex-col justify-center shadow-sm">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Active Incidents</span>
          <span className="text-lg font-mono text-blue-400 mt-0.5">{kpis.activeIncidents}</span>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg flex flex-col justify-center shadow-sm">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Critical</span>
          <span className="text-lg font-mono text-red-400 mt-0.5">{kpis.criticalIncidents}</span>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg flex flex-col justify-center shadow-sm">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">En Route</span>
          <span className="text-lg font-mono text-blue-300 mt-0.5">{kpis.enRoute}</span>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg flex flex-col justify-center shadow-sm">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">At Scene</span>
          <span className="text-lg font-mono text-orange-400 mt-0.5">{kpis.atScene}</span>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg flex flex-col justify-center shadow-sm">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">In Transit</span>
          <span className="text-lg font-mono text-purple-400 mt-0.5">{kpis.inTransit}</span>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg flex flex-col justify-center shadow-sm">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Available</span>
          <span className="text-lg font-mono text-emerald-400 mt-0.5">{kpis.available}</span>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg flex flex-col justify-center shadow-sm">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Hosp Receiving</span>
          <span className="text-lg font-mono text-teal-400 mt-0.5">{kpis.receivingHospitals}</span>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg flex flex-col justify-center shadow-sm">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Attention Req</span>
          <span className={`text-lg font-mono mt-0.5 ${kpis.attentionReq > 0 ? 'text-red-500' : 'text-slate-300'}`}>{kpis.attentionReq}</span>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg flex flex-col justify-center shadow-sm">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Avg Resp Time</span>
          <span className="text-[10px] font-mono text-slate-600 mt-2 font-bold tracking-widest">UNAVAILABLE</span>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 p-3 rounded-lg flex flex-col justify-center shadow-sm">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Avg Hosp ETA</span>
          <span className="text-lg font-mono text-emerald-300 mt-0.5">{kpis.avgEta != null ? `${kpis.avgEta}m` : '-'}</span>
        </div>
      </div>

      <div className="flex-1 overflow-hidden flex flex-col xl:flex-row gap-5 pb-4">
        
        {/* LEFT COLUMN: Data Tables */}
        <div className="flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-6 pr-2">
          
          <div className="flex flex-col lg:flex-row gap-5 h-[350px] shrink-0">
            {/* Incidents Table */}
            <div className="flex-1 flex flex-col bg-slate-900/50 rounded-xl border border-slate-800 overflow-hidden shadow-sm">
              <div className="p-3 border-b border-slate-800 bg-slate-900/80 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 shrink-0">
                <h3 className="font-bold text-xs uppercase tracking-widest text-slate-200">Emergency Incident Operations</h3>
                <div className="flex flex-wrap gap-2">
                  <input type="text" placeholder="Search ID, Location..." value={incidentSearch} onChange={(e) => setIncidentSearch(e.target.value)} className="bg-slate-950 border border-slate-800 text-[10px] rounded px-2 py-1 outline-none text-slate-200 w-[150px]"/>
                  <select value={incidentPriority} onChange={(e) => setIncidentPriority(e.target.value)} className="bg-slate-950 border border-slate-800 text-[10px] rounded px-2 py-1 outline-none text-slate-300">
                    <option value="All">All Priorities</option><option value="Critical">Critical</option><option value="High">High</option><option value="Moderate">Moderate</option>
                  </select>
                </div>
              </div>
              
              <div className="overflow-x-auto custom-scrollbar flex-1 bg-slate-950">
                <table className="w-full text-left text-sm whitespace-nowrap min-w-[900px]">
                  <thead className="bg-slate-950/90 text-[9px] uppercase tracking-widest text-slate-500 sticky top-0 border-b border-slate-800 z-10">
                    <tr>
                      <th className="px-4 py-3 font-bold">Incident ID</th>
                      <th className="px-4 py-3 font-bold">Type</th>
                      <th className="px-4 py-3 font-bold">Location</th>
                      <th className="px-4 py-3 font-bold">Priority</th>
                      <th className="px-4 py-3 font-bold">Status</th>
                      <th className="px-4 py-3 font-bold">Patients</th>
                      <th className="px-4 py-3 font-bold">Amb (Req/Assg)</th>
                      <th className="px-4 py-3 font-bold">Receiving Hospital</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/50">
                    {filteredIncidents.map((inc) => {
                      const isSelected = selectedIncident?.incident_id === inc.incident_id;
                      const isCritical = inc.priority === 'CRITICAL';
                      return (
                        <tr key={inc.incident_id} onClick={() => { setSelectedIncident(inc); setSelectedAmbulance(null); }} className={`cursor-pointer transition-colors ${isSelected ? 'bg-blue-900/20' : 'hover:bg-slate-800/50'} ${inc.isSimulated ? 'bg-indigo-900/10' : ''}`}>
                          <td className={`px-4 py-3 font-bold text-slate-200 border-l-2 ${isSelected ? 'border-blue-500' : isCritical ? 'border-red-500' : 'border-transparent'}`}>{inc.incident_id} {inc.isSimulated && <span className="ml-2 text-[8px] bg-indigo-500/20 text-indigo-400 px-1 py-0.5 rounded border border-indigo-500/50">SIM</span>}</td>
                          <td className="px-4 py-3 text-xs text-slate-300">{inc.incident_type}</td>
                          <td className="px-4 py-3 font-mono text-xs text-slate-400">{inc.location}</td>
                          <td className="px-4 py-3"><span className={`px-2 py-0.5 text-[9px] font-bold uppercase tracking-widest rounded border ${isCritical ? 'bg-red-900/20 text-red-400 border-red-800/50' : inc.priority === 'HIGH' ? 'bg-orange-900/20 text-orange-400 border-orange-800/50' : 'bg-slate-800/50 text-slate-400 border-slate-700/50'}`}>{inc.priority}</span></td>
                          <td className="px-4 py-3 text-[10px] font-bold uppercase tracking-wider text-slate-400">{inc.incident_status?.replace(/_/g, ' ')}</td>
                          <td className="px-4 py-3 font-mono text-xs text-orange-300">{inc.patient_count ?? 'N/A'}</td>
                          <td className="px-4 py-3 font-mono text-xs text-blue-300">{inc.ambulances_required ?? '-'}/{inc.ambulances_assigned ?? '-'}</td>
                          <td className="px-4 py-3 text-xs text-emerald-400 truncate max-w-[150px]">{inc.receiving_hospital_name || 'PENDING'}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="flex flex-col lg:flex-row gap-5 flex-1 min-h-[300px]">
            {/* Ambulances Table */}
            <div className="flex-1 flex flex-col bg-slate-900/50 rounded-xl border border-slate-800 overflow-hidden shadow-sm">
              <div className="p-3 border-b border-slate-800 bg-slate-900/80 flex justify-between items-center shrink-0">
                <h3 className="font-bold text-xs uppercase tracking-widest text-slate-200">Active Fleet Operations</h3>
              </div>
              <div className="overflow-y-auto custom-scrollbar flex-1 bg-slate-950">
                <table className="w-full text-left text-sm whitespace-nowrap">
                  <thead className="bg-slate-950/90 text-[9px] uppercase tracking-widest text-slate-500 sticky top-0 border-b border-slate-800 z-10">
                    <tr>
                      <th className="px-4 py-3 font-bold">Unit ID</th>
                      <th className="px-4 py-3 font-bold">Incident</th>
                      <th className="px-4 py-3 font-bold">Status</th>
                      <th className="px-4 py-3 font-bold">Destination</th>
                      <th className="px-4 py-3 font-bold text-right">ETA</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/50">
                    {displayAmbulances.map((unit) => {
                      const isCritical = unit.priority === 'CRITICAL';
                      const isSelected = selectedAmbulance?.ambulance_id === unit.ambulance_id;
                      let statusColor = "text-slate-400";
                      if (unit.status.includes('TRANSPORT')) statusColor = "text-purple-400";
                      else if (unit.status.includes('ROUTE') || unit.status.includes('APPROACH')) statusColor = "text-blue-400";
                      else if (unit.status.includes('SCENE') || unit.status.includes('HANDOVER')) statusColor = "text-orange-400";
                      else if (unit.status.includes('AVAILABLE') || unit.status.includes('RETURNING')) statusColor = "text-emerald-400";

                      return (
                        <tr key={unit.ambulance_id} onClick={() => { setSelectedAmbulance(unit); setSelectedIncident(null); }} className={`cursor-pointer transition-colors ${isSelected ? 'bg-blue-900/20 hover:bg-blue-900/30' : 'hover:bg-slate-800/50'} ${unit.isSimulated ? 'bg-indigo-900/10' : ''}`}>
                          <td className={`px-4 py-3 font-bold text-slate-200 border-l-2 ${isSelected ? 'border-blue-500' : isCritical ? 'border-red-500' : 'border-transparent'}`}>{unit.ambulance_id} {unit.isSimulated && <span className="ml-2 text-[8px] bg-indigo-500/20 text-indigo-400 px-1 py-0.5 rounded border border-indigo-500/50">SIM</span>}</td>
                          <td className="px-4 py-3 font-mono text-[10px] text-slate-400">{unit.incident_id || 'N/A'}</td>
                          <td className={`px-4 py-3 text-[10px] font-bold uppercase tracking-wider ${statusColor}`}>{unit.status.replace(/_/g, ' ')}</td>
                          <td className="px-4 py-3 text-slate-300 font-bold text-xs truncate max-w-[150px]">{unit.recommended_hospital || '-'}</td>
                          <td className="px-4 py-3 font-mono text-blue-400 text-right">{unit.eta_minutes != null ? `${unit.eta_minutes}m` : '-'}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* PHASE 6: GLOBAL EVENT FEED */}
            <div className="w-[400px] flex flex-col bg-slate-900/50 rounded-xl border border-slate-800 overflow-hidden shadow-sm shrink-0">
              <div className="p-3 border-b border-slate-800 bg-slate-900/80 flex justify-between items-center shrink-0">
                <h3 className="font-bold text-xs uppercase tracking-widest text-slate-200">Global Operational Event Feed</h3>
              </div>
              <div className="overflow-y-auto custom-scrollbar flex-1 bg-slate-950 p-4">
                {activeSimData.logs.length > 0 ? (
                  <div className="flex flex-col gap-4">
                    {activeSimData.logs.map((log, idx) => (
                      <div key={idx} className="flex flex-col gap-1 border-b border-slate-800/50 pb-3 last:border-0 last:pb-0">
                        <div className="flex justify-between items-center">
                          <span className={`text-[10px] font-bold uppercase tracking-widest ${idx === 0 ? 'text-blue-400' : 'text-slate-500'}`}>{log.state.replace(/_/g, ' ')}</span>
                          <span className="text-[9px] text-slate-500 font-mono">{log.timestamp}</span>
                        </div>
                        <span className={`text-xs ${idx === 0 ? 'text-slate-200 font-medium' : 'text-slate-400'}`}>{log.text}</span>
                        <div className="flex gap-2 mt-1">
                          <span className="text-[9px] bg-slate-800/50 text-slate-400 px-1.5 py-0.5 rounded border border-slate-700/50 font-mono">{log.incidentId}</span>
                          <span className="text-[9px] bg-slate-800/50 text-slate-400 px-1.5 py-0.5 rounded border border-slate-700/50 font-mono">{log.ambulanceId}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center text-slate-500 text-sm py-8 italic font-mono">No recent events.</div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: SIMULATION CONTROL CENTER / DETAILS */}
        <div className={`w-full xl:w-[450px] shrink-0 h-full overflow-hidden bg-slate-900/80 border border-slate-800 rounded-xl shadow-sm relative flex flex-col transition-all`}>
          
          {isSimMode ? (
            /* SIMULATION CONTROL CENTER */
            <div className="p-5 flex flex-col h-full overflow-hidden relative">
              
              <div className="border-b border-indigo-500/30 pb-4 mb-4 flex justify-between items-start shrink-0">
                <div>
                  <div className="text-[9px] text-indigo-400 font-mono tracking-widest uppercase mb-1 flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(99,102,241,0.8)]"></span> SIMULATION CONTROL CENTER
                  </div>
                  <h2 className="text-xl font-bold text-white uppercase tracking-tight">Emergency Simulator</h2>
                  <div className="text-[10px] text-slate-400 font-mono mt-1 uppercase tracking-wider">Deterministic Multi-Agent Validation Engine</div>
                </div>
              </div>

              <div className="flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-5">
                
                {/* Scenario Selection & Overrides */}
                <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                  <div className="mb-4">
                    <label className="text-[9px] text-slate-500 uppercase tracking-widest font-bold block mb-1.5">Predefined Scenario</label>
                    <select 
                      value={selectedScenarioId} 
                      onChange={(e) => { setSelectedScenarioId(e.target.value); setSimStatus('READY'); setSimTime(0); setSimAgentIntel(null); }}
                      disabled={simStatus !== 'READY'}
                      className="w-full bg-slate-900 border border-slate-700 text-xs rounded px-3 py-2 outline-none text-slate-200 disabled:opacity-50 transition-colors"
                    >
                      {PREDEFINED_SCENARIOS.map(s => <option key={s.id} value={s.id}>{s.name} ({s.type})</option>)}
                    </select>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-[9px] text-slate-500 uppercase tracking-widest font-bold block mb-1.5">Severity</label>
                      <select 
                        value={simOverrides.severity || activeSimConfig.severity} 
                        onChange={(e) => setSimOverrides(prev => ({...prev, severity: e.target.value}))}
                        disabled={simStatus !== 'READY'}
                        className="w-full bg-slate-900 border border-slate-700 text-xs rounded px-2 py-1.5 outline-none text-slate-200 disabled:opacity-50"
                      >
                        <option value="MODERATE">Moderate</option>
                        <option value="HIGH">High</option>
                        <option value="CRITICAL">Critical</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-[9px] text-slate-500 uppercase tracking-widest font-bold block mb-1.5">Ambulances Required</label>
                      <input 
                        type="number" min="1" max="20"
                        value={simOverrides.ambs !== '' ? simOverrides.ambs : activeSimConfig.ambs} 
                        onChange={(e) => setSimOverrides(prev => ({...prev, ambs: e.target.value}))}
                        disabled={simStatus !== 'READY'}
                        className="w-full bg-slate-900 border border-slate-700 text-xs rounded px-2 py-1.5 outline-none text-slate-200 disabled:opacity-50"
                      />
                    </div>
                    <div>
                      <label className="text-[9px] text-slate-500 uppercase tracking-widest font-bold block mb-1.5">Total Patients</label>
                      <input 
                        type="number" min="1" max="100"
                        value={simOverrides.patients !== '' ? simOverrides.patients : activeSimConfig.patients} 
                        onChange={(e) => setSimOverrides(prev => ({...prev, patients: e.target.value}))}
                        disabled={simStatus !== 'READY'}
                        className="w-full bg-slate-900 border border-slate-700 text-xs rounded px-2 py-1.5 outline-none text-slate-200 disabled:opacity-50"
                      />
                    </div>
                    <div>
                      <label className="text-[9px] text-slate-500 uppercase tracking-widest font-bold block mb-1.5">Sim Speed</label>
                      <select 
                        value={simSpeed} 
                        onChange={(e) => setSimSpeed(Number(e.target.value))}
                        className="w-full bg-slate-900 border border-indigo-500/50 text-xs rounded px-2 py-1.5 outline-none text-indigo-300 bg-indigo-900/20"
                      >
                        <option value={0.5}>0.5x (Slow)</option>
                        <option value={1}>1.0x (Normal)</option>
                        <option value={2}>2.0x (Fast)</option>
                        <option value={5}>5.0x (Rapid)</option>
                      </select>
                    </div>
                  </div>
                </div>

                {/* Controls */}
                <div className="flex gap-2 shrink-0">
                  {simStatus === 'READY' || simStatus === 'COMPLETE' ? (
                    <button onClick={() => handleSimAction('START')} className="flex-1 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold uppercase tracking-widest rounded transition-colors shadow-sm">
                      Start Scenario
                    </button>
                  ) : simStatus === 'RUNNING' ? (
                    <button onClick={() => handleSimAction('PAUSE')} className="flex-1 py-2.5 bg-orange-600/20 hover:bg-orange-600/40 text-orange-400 border border-orange-500/50 text-xs font-bold uppercase tracking-widest rounded transition-colors shadow-sm">
                      Pause
                    </button>
                  ) : (
                    <button onClick={() => handleSimAction('RESUME')} className="flex-1 py-2.5 bg-emerald-600/20 hover:bg-emerald-600/40 text-emerald-400 border border-emerald-500/50 text-xs font-bold uppercase tracking-widest rounded transition-colors shadow-sm">
                      Resume
                    </button>
                  )}
                  <button onClick={() => handleSimAction('RESET')} disabled={simStatus === 'READY'} className="px-6 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-bold uppercase tracking-widest rounded border border-slate-700 disabled:opacity-50 transition-colors shadow-sm">
                    Reset
                  </button>
                </div>

                {/* Progress & Current State */}
                <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-800 shrink-0">
                  <div className="flex justify-between items-center text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2.5">
                    <span>State: <span className="text-indigo-400 ml-1">{activeSimData.currentState.replace(/_/g, ' ')}</span></span>
                    <span className="font-mono text-slate-300">T+{simTime}s / {activeSimConfig.duration}s</span>
                  </div>
                  <div className="h-2.5 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                    <div className="h-full bg-indigo-500 transition-all duration-300 ease-linear shadow-[0_0_8px_rgba(99,102,241,0.8)]" style={{ width: `${Math.min((simTime / activeSimConfig.duration) * 100, 100)}%` }}></div>
                  </div>
                </div>

                {/* Phase 7: AGENT DECISION FEED */}
                {simAgentIntel && (
                  <div className="flex flex-col gap-3 mt-1">
                    <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest border-b border-slate-800/50 pb-1">Agent Decision Feed</h3>
                    
                    <div className="bg-blue-950/20 border border-blue-900/40 p-3 rounded-lg flex flex-col gap-1">
                      <div className="flex justify-between items-center text-[9px] font-mono text-slate-500 mb-1">
                        <span>T+2s • HOSPITAL AGENT</span>
                        <span className="text-blue-400 font-bold border border-blue-900 px-1 rounded">Conf: {simAgentIntel.hospital.confidence}%</span>
                      </div>
                      <div className="text-xs text-slate-200 font-bold uppercase">{simAgentIntel.hospital.decision_type || 'CAPACITY_ASSESSMENT'}</div>
                      <div className="text-[11px] text-slate-400 leading-snug">{simAgentIntel.hospital.hospital_name || activeSimConfig.dest} recommended. {simAgentIntel.hospital.expected_impact}</div>
                    </div>

                    <div className="bg-emerald-950/20 border border-emerald-900/40 p-3 rounded-lg flex flex-col gap-1">
                      <div className="flex justify-between items-center text-[9px] font-mono text-slate-500 mb-1">
                        <span>T+3s • TRAFFIC AGENT</span>
                        <span className="text-emerald-400 font-bold border border-emerald-900 px-1 rounded">Conf: {simAgentIntel.traffic.confidence}%</span>
                      </div>
                      <div className="text-xs text-slate-200 font-bold uppercase">ROUTE ASSESSMENT</div>
                      <div className="text-[11px] text-slate-400 leading-snug">{simAgentIntel.traffic.recommended_route}. ETA: {simAgentIntel.traffic.estimated_travel_time}.</div>
                    </div>
                  </div>
                )}

              </div>
            </div>
          ) : (
            /* STANDARD DETAILS PANEL (Incidents or Ambulances) */
            <div className="p-5 flex flex-col h-full overflow-y-auto custom-scrollbar">
              {selectedIncident ? (
                <>
                  <div className="border-b border-slate-800/80 pb-3 mb-4 flex justify-between items-start shrink-0">
                    <div>
                      <div className="text-[9px] text-blue-400 font-mono tracking-widest uppercase mb-1">INCIDENT OVERVIEW</div>
                      <h2 className="text-lg font-bold text-white uppercase tracking-tight">{selectedIncident.incident_id}</h2>
                      <div className="text-xs text-slate-400 mt-1 font-semibold uppercase">{selectedIncident.incident_type}</div>
                    </div>
                    <button onClick={() => setSelectedIncident(null)} className="bg-slate-800/50 hover:bg-slate-700 text-slate-400 p-1.5 rounded-lg border border-slate-700/50 transition-colors"><svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg></button>
                  </div>

                  <div className="grid grid-cols-2 gap-2 mb-4 shrink-0">
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800/80">
                       <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold block mb-1">Location</span>
                       <span className="text-xs font-mono text-slate-300 truncate block">{selectedIncident.location}</span>
                    </div>
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800/80">
                       <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold block mb-1">Priority</span>
                       <span className={`text-xs font-bold uppercase tracking-wider ${selectedIncident.priority === 'CRITICAL' ? 'text-red-400' : 'text-orange-400'}`}>{selectedIncident.priority}</span>
                    </div>
                  </div>

                  <div className="flex flex-col gap-4 flex-1">
                    <div>
                      <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2 border-b border-slate-800/50 pb-1 flex justify-between">Patient Overview <span className="text-[8px] bg-slate-800 px-1 rounded text-slate-500">(SIMULATED)</span></h3>
                      <div className="grid grid-cols-4 gap-2 text-center">
                        <div className="bg-slate-950 p-2 rounded border border-slate-800/50"><span className="text-[9px] text-slate-500 block mb-1">Total</span><span className="text-sm font-bold text-slate-200">{selectedIncident.patient_count ?? 'N/A'}</span></div>
                        <div className="bg-slate-950 p-2 rounded border border-slate-800/50"><span className="text-[9px] text-slate-500 block mb-1">Critical</span><span className="text-sm font-bold text-red-400">{selectedIncident.critical_patient_count ?? 'N/A'}</span></div>
                        <div className="bg-slate-950 p-2 rounded border border-slate-800/50"><span className="text-[9px] text-slate-500 block mb-1">Serious</span><span className="text-sm font-bold text-orange-400">{selectedIncident.serious_patient_count ?? 'N/A'}</span></div>
                        <div className="bg-slate-950 p-2 rounded border border-slate-800/50"><span className="text-[9px] text-slate-500 block mb-1">Mod</span><span className="text-sm font-bold text-yellow-400">{(selectedIncident.patient_count && selectedIncident.critical_patient_count != null) ? Math.max(0, selectedIncident.patient_count - selectedIncident.critical_patient_count - (selectedIncident.serious_patient_count || 0)) : 'N/A'}</span></div>
                      </div>
                    </div>
                    <div>
                      <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2 border-b border-slate-800/50 pb-1">Response</h3>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="bg-slate-950 p-2.5 rounded border border-slate-800/50 flex justify-between items-center"><span className="text-[9px] text-slate-500 uppercase tracking-widest">Amb Required</span><span className="text-xs font-mono text-slate-300">{selectedIncident.ambulances_required ?? 'N/A'}</span></div>
                        <div className="bg-slate-950 p-2.5 rounded border border-slate-800/50 flex justify-between items-center"><span className="text-[9px] text-slate-500 uppercase tracking-widest">Amb Assigned</span><span className="text-xs font-mono text-blue-400 font-bold">{selectedIncident.ambulances_assigned ?? 'N/A'}</span></div>
                        <div className="bg-slate-950 p-2.5 rounded border border-slate-800/50 flex justify-between items-center col-span-2"><span className="text-[9px] text-slate-500 uppercase tracking-widest">Response Status</span><span className="text-xs font-bold text-slate-300 uppercase">{selectedIncident.incident_status?.replace(/_/g, ' ') || 'N/A'}</span></div>
                      </div>
                    </div>
                    <div>
                      <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2 border-b border-slate-800/50 pb-1">Receiving Hospital</h3>
                      <div className="bg-slate-950 p-3 rounded-lg border border-slate-800/80 flex flex-col gap-2">
                        <div className="flex justify-between items-center"><span className="text-[10px] font-bold text-slate-200">{selectedIncident.receiving_hospital_name || 'PENDING ASSIGNMENT'}</span></div>
                        <div className="flex justify-between items-center text-[9px] font-mono border-t border-slate-800/50 pt-2"><span className="text-slate-500 uppercase">Emergency Status</span><span className={selectedIncident.hospital_emergency_status === 'CRITICAL' ? 'text-red-400' : 'text-emerald-400'}>{selectedIncident.hospital_emergency_status || 'UNKNOWN'}</span></div>
                        <div className="flex justify-between items-center text-[9px] font-mono"><span className="text-slate-500 uppercase">ETA</span><span className="text-orange-400">{selectedIncident.hospital_eta ? new Date(selectedIncident.hospital_eta).toLocaleTimeString() : 'N/A'}</span></div>
                      </div>
                    </div>

                    {/* PHASE 6: DETAIL TIMELINE SECTION */}
                    <div>
                      <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3 border-b border-slate-800/50 pb-1">Operational Timeline</h3>
                      <div className="pl-2 pt-2 pb-2">
                        {selectedIncident.isSimulated && activeSimData.logs.length > 0 ? (
                          activeSimData.logs.map((log, idx) => (
                            <TimelineEvent key={idx} event={log} isNewest={idx === 0} />
                          ))
                        ) : (
                          <div className="text-[10px] text-slate-500 font-mono italic">No timeline events available from backend.</div>
                        )}
                      </div>
                    </div>

                  </div>
                </>
              ) : selectedAmbulance ? (
                <>
                  <div className="border-b border-slate-800/80 pb-4 mb-4 flex justify-between items-start shrink-0">
                    <div>
                      <div className="text-[9px] text-blue-400 font-mono tracking-widest uppercase mb-1">UNIT INTELLIGENCE</div>
                      <h2 className="text-xl font-bold text-white uppercase tracking-tight">{selectedAmbulance.ambulance_id}</h2>
                      <div className="flex gap-2 text-[10px] text-slate-400 font-mono mt-1.5 uppercase tracking-wider">
                        <span>{selectedAmbulance.equipment_level || 'N/A'}</span><span className="text-slate-600">•</span><span>{selectedAmbulance.crew_type || 'N/A'}</span>
                      </div>
                    </div>
                    <button onClick={() => setSelectedAmbulance(null)} className="bg-slate-800/50 hover:bg-slate-700 text-slate-400 p-1.5 rounded-lg border border-slate-700/50"><svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg></button>
                  </div>
                  <div className="grid grid-cols-2 gap-3 mb-5 shrink-0">
                    <div className="bg-slate-950 p-3 rounded-lg border border-slate-800/80">
                       <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold block mb-1">Status</span>
                       <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">{selectedAmbulance.status.replace(/_/g, ' ')}</span>
                    </div>
                    <div className="bg-slate-950 p-3 rounded-lg border border-slate-800/80">
                       <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold block mb-1">ETA</span>
                       <span className="text-lg font-mono font-bold text-blue-400">{selectedAmbulance.eta_minutes != null ? `${selectedAmbulance.eta_minutes} min` : 'N/A'}</span>
                    </div>
                  </div>
                  <div className="flex flex-col gap-5 flex-1">
                    <div>
                      <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3 border-b border-slate-800/50 pb-1">Destination Intelligence</h3>
                      <div className="bg-slate-950 p-4 rounded-lg border border-slate-800/80 flex flex-col gap-3">
                        <div className="flex justify-between items-center mb-1"><span className="text-sm font-bold text-slate-200">{selectedAmbulance.recommended_hospital || 'Awaiting Routing'}</span><span className="text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded uppercase font-bold tracking-widest">Destination</span></div>
                        <div className="flex justify-between items-center text-[10px] font-mono border-t border-slate-800/50 pt-2 mt-1"><span className="text-slate-500 uppercase tracking-widest">Capacity Status</span><span className={selectedAmbulance.hospital_capacity_status === 'CRITICAL' ? 'text-red-400 font-bold' : 'text-emerald-400'}>{selectedAmbulance.hospital_capacity_status || 'UNKNOWN'}</span></div>
                        <div className="flex justify-between items-center text-[10px] font-mono border-t border-slate-800/50 pt-2"><span className="text-slate-500 uppercase tracking-widest">Emergency Load</span><span className={selectedAmbulance.hospital_emergency_status === 'CRITICAL' ? 'text-red-400 font-bold' : 'text-orange-400'}>{selectedAmbulance.hospital_emergency_status || 'UNKNOWN'}</span></div>
                      </div>
                    </div>

                    {/* PHASE 6: DETAIL TIMELINE SECTION */}
                    <div>
                      <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3 border-b border-slate-800/50 pb-1">Operational Timeline</h3>
                      <div className="pl-2 pt-2 pb-2">
                        {selectedAmbulance.isSimulated && activeSimData.logs.length > 0 ? (
                          activeSimData.logs.map((log, idx) => (
                            <TimelineEvent key={idx} event={log} isNewest={idx === 0} />
                          ))
                        ) : (
                          <div className="text-[10px] text-slate-500 font-mono italic">No timeline events available from backend.</div>
                        )}
                      </div>
                    </div>

                    <div className="mt-auto flex flex-col gap-3 shrink-0 pt-4">
                      {/* PHASE 7: Replaced hardcoded text with backend simulation recommendation (or fallback standard message) */}
                      <div className="bg-blue-950/20 border border-blue-900/40 p-4 rounded-xl shadow-sm">
                        <span className="text-[9px] text-blue-400 uppercase tracking-widest font-bold flex items-center gap-2 mb-2"><span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse"></span>SIMULATION RECOMMENDATION: HOSPITAL AGENT</span>
                        <div className="text-xs text-slate-300 leading-relaxed font-medium">
                          {selectedAmbulance.isSimulated && simAgentIntel ? 
                            simAgentIntel.hospital.recommendation : 
                            'Recommend maintaining current destination routing. Required specialist care teams have been placed on standby.'}
                        </div>
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <div className="flex flex-col items-center justify-center text-slate-500 italic h-full gap-2">
                  <svg className="w-8 h-8 text-slate-700" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                  <span className="text-xs font-mono">Select an incident to view intelligence.</span>
                </div>
              )}
            </div>
          )}
        </div>

      </div>
    </div>
  );
}