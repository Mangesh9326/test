import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

const LOCATIONS = [
  { id:'CAM-101', name:'Ruia College Road', road:'Matunga West', lat:19.023845829174906, lng:72.8496884047973 },
  { id:'CAM-102', name:'Dadar Railway Station', road:'Dr. Ambedkar Rd', lat:19.018, lng:72.8436 },
  { id:'CAM-103', name:'Wadala Highway (Eastern Freeway Junction)', road:'Eastern Freeway', lat:19.0218, lng:72.8745 },
];
const show = value => value === null || value === undefined || value === '' ? 'N/A' : value;

export default function SimulateDemonstation() {
  const mounted = useRef(true); const startLock = useRef(false);
  const [selectedCamera,setSelectedCamera] = useState(LOCATIONS[0]); const [scenarios,setScenarios] = useState([]); const [selectedScenarioId,setSelectedScenarioId] = useState(''); const [scenarioDetails,setScenarioDetails] = useState(null);
  const [simulationStatus,setSimulationStatus] = useState('IDLE'); const [createdIncidentId,setCreatedIncidentId] = useState(null); const [startResponse,setStartResponse] = useState(null); const [trafficError,setTrafficError] = useState(null);
  const [activeIncidents,setActiveIncidents] = useState([]); const [selectedIncidentId,setSelectedIncidentId] = useState(''); const [hospitalSuccess,setHospitalSuccess] = useState(false); const [hospitalError,setHospitalError] = useState(null); const [hospitalLoading,setHospitalLoading] = useState(false);
  const refreshActiveIncidents = useCallback(async (preferredId) => {
    const res = await fetch('/api/hospitals/incidents');
    if (!res.ok) throw new Error('Unable to refresh active incidents.');
    const data = await res.json(); const incidents = Array.isArray(data) ? data : [];
    if (!mounted.current) return [];
    setActiveIncidents(incidents);
    const target = preferredId || createdIncidentId;
    if (target && incidents.some(i => String(i.incident_id) === String(target))) setSelectedIncidentId(String(target));
    return incidents;
  }, [createdIncidentId]);
  useEffect(() => { mounted.current=true; return () => { mounted.current=false; }; }, []);
  useEffect(() => { fetch('/api/simulation/scenario').then(r=>r.ok?r.json():Promise.reject()).then(data=>mounted.current&&setScenarios((Array.isArray(data)?data:data.scenarios||[]).filter(s=>s.id!=='live'))).catch(()=>mounted.current&&setTrafficError('Unable to load simulation scenarios.')); }, []);
  useEffect(() => { if(!selectedScenarioId){queueMicrotask(()=>setScenarioDetails(null));return;} fetch(`/api/simulation/scenario/${selectedScenarioId}`).then(r=>r.ok?r.json():Promise.reject()).then(data=>mounted.current&&setScenarioDetails(data.scenario||data)).catch(()=>mounted.current&&setScenarioDetails(null)); }, [selectedScenarioId]);
  useEffect(() => { const initial=setTimeout(()=>refreshActiveIncidents().catch(()=>{}),0); const id=setInterval(()=>refreshActiveIncidents().catch(()=>{}),8000); return()=>{clearTimeout(initial);clearInterval(id);}; }, [refreshActiveIncidents]);
  const selectedIncident = useMemo(() => activeIncidents.find(i=>String(i.incident_id)===String(selectedIncidentId)), [activeIncidents,selectedIncidentId]);
  const evaluateHospitalAgent = async (incidentId, incident) => {
    setHospitalLoading(true); setHospitalError(null); setHospitalSuccess(false);
    const body = { incident_id:incidentId, incident_type:incident?.incident_type, location:incident?.location, severity:incident?.severity, destination_hospital:incident?.receiving_hospital_name||incident?.receiving_hospital_id };
    Object.keys(body).forEach(key => body[key] === undefined && delete body[key]);
    try { const res=await fetch('/api/hospitals/agent/evaluate-emergency',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); const data=await res.json().catch(()=>({})); if(!res.ok) throw new Error(data.error||data.message||'Incident started successfully, but Hospital Agent evaluation failed.'); if(mounted.current)setHospitalSuccess(true); }
    catch(error){if(mounted.current)setHospitalError(error.message||'Incident started successfully, but Hospital Agent evaluation failed.');}
    finally { if(mounted.current)setHospitalLoading(false); }
  };
  const handleStartSimulation = async () => {
    if(startLock.current || simulationStatus==='RUNNING') return;
    if(!selectedScenarioId){setTrafficError('Select an emergency scenario before starting.');return;}
    if(!selectedCamera || !Number.isFinite(Number(selectedCamera.lat)) || !Number.isFinite(Number(selectedCamera.lng))){setTrafficError('Selected location has invalid coordinates.');return;}
    startLock.current=true; setSimulationStatus('STARTING'); setTrafficError(null); setHospitalError(null); setHospitalSuccess(false); setCreatedIncidentId(null); setSelectedIncidentId('');
    try {
      const res=await fetch('/api/simulation/start-inference',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cameraId:selectedCamera.id,locationName:selectedCamera.name,scenarioId:selectedScenarioId,lat:selectedCamera.lat,lng:selectedCamera.lng})});
      const data=await res.json().catch(()=>({})); if(!res.ok||!data.success) throw new Error(data.error||data.message||'Simulation server could not start the selected scenario.'); if(!data.incidentId) throw new Error('Simulation request succeeded but the backend did not return an incident ID.');
      if(Number(data.latitude)!==Number(selectedCamera.lat)||Number(data.longitude)!==Number(selectedCamera.lng)) throw new Error('Simulation started, but the backend returned coordinates different from the selected location.');
      setCreatedIncidentId(data.incidentId); setSelectedIncidentId(String(data.incidentId)); setStartResponse(data); const incidents=await refreshActiveIncidents(data.incidentId); const incident=incidents.find(i=>String(i.incident_id)===String(data.incidentId)); setSimulationStatus('RUNNING');
      if(!incident) setHospitalError('Simulation started, but the created incident could not be synchronized. Refresh incidents to retry synchronization.'); else evaluateHospitalAgent(data.incidentId,incident);
    } catch(error) { setTrafficError(error.message||'Simulation server could not start the selected scenario.'); setSimulationStatus('FAILED'); }
    finally { startLock.current=false; }
  };
  const reset = () => { startLock.current=false; setSimulationStatus('IDLE'); setCreatedIncidentId(null); setSelectedIncidentId(''); setStartResponse(null); setTrafficError(null); setHospitalError(null); setHospitalSuccess(false); };
  const statusColor = simulationStatus==='RUNNING'?'text-emerald-400':simulationStatus==='STARTING'?'text-blue-400':simulationStatus==='FAILED'?'text-red-400':'text-slate-400';
  return <div className="h-dvh overflow-y-auto bg-slate-950 text-slate-200"><main className="max-w-7xl mx-auto p-4 sm:p-6 lg:p-8 space-y-6">
    <header className="border-b border-slate-800 pb-5"><span className="text-[10px] font-bold tracking-widest text-blue-400">DEMONSTRATION MODE</span><h1 className="mt-2 text-2xl font-bold text-white uppercase">Operations Control Center</h1><p className="text-sm text-slate-400">Simulation Launchpad</p></header>
    <div className="grid xl:grid-cols-2 gap-6">
      <section className="bg-slate-900 border border-slate-700 rounded-xl overflow-hidden"><div className="p-5 border-b border-slate-700 bg-slate-800/30"><p className="text-[10px] font-bold text-blue-400 tracking-widest">STEP 1</p><h2 className="mt-1 font-bold uppercase">Traffic / Incident Simulation</h2><p className="mt-1 text-xs text-slate-400">Launch a scenario at one operational location. The created incident is used directly by hospital response.</p></div><div className="p-5 space-y-6">
        <div><label className="block mb-2 text-[10px] font-bold uppercase tracking-widest text-slate-500">1. Operational Location / Camera</label><div className="grid sm:grid-cols-3 gap-3">{LOCATIONS.map(camera=><button key={camera.id} disabled={simulationStatus==='STARTING'||simulationStatus==='RUNNING'} onClick={()=>setSelectedCamera(camera)} className={`p-3 text-left rounded border disabled:opacity-50 ${selectedCamera.id===camera.id?'border-blue-500 bg-blue-500/10':'border-slate-700 bg-slate-950'}`}><p className="font-mono text-xs font-bold">{camera.id}</p><p className="mt-1 text-xs">{camera.name}</p><p className="mt-2 text-[10px] text-emerald-400">{camera.lat}, {camera.lng}</p></button>)}</div></div>
        <div><label className="block mb-2 text-[10px] font-bold uppercase tracking-widest text-slate-500">2. Database Scenario Feed</label><select value={selectedScenarioId} disabled={simulationStatus==='STARTING'||simulationStatus==='RUNNING'} onChange={e=>setSelectedScenarioId(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded px-3 py-3 text-sm"><option value="">-- Select Emergency Scenario --</option>{scenarios.map(s=><option key={s.id} value={s.id}>{s.name}</option>)}</select></div>
        <div className="bg-slate-950 border border-slate-800 rounded p-4"><p className="mb-3 pb-2 border-b border-slate-800 text-[10px] font-bold uppercase tracking-widest text-slate-500">Incident Preview</p>{scenarioDetails?<div className="grid grid-cols-2 gap-3 text-xs"><Info label="Scenario" value={scenarioDetails.name}/><Info label="Type" value={scenarioDetails.incident_type||scenarioDetails.type}/><Info label="Severity" value={scenarioDetails.severity}/><Info label="Location" value={selectedCamera.name}/><Info label="Coordinates" value={`${selectedCamera.lat}, ${selectedCamera.lng}`}/></div>:<p className="text-xs text-slate-500">Select a scenario to preview the incident.</p>}</div>
        {trafficError&&<Notice color="red" text={trafficError}/>} {simulationStatus==='STARTING'&&<Notice color="blue" text="SIMULATION STARTING — creating the incident and synchronizing hospital response."/>}
        {simulationStatus==='RUNNING'&&<div className="bg-emerald-950/20 border border-emerald-700/50 rounded p-4 space-y-2"><p className="text-xs font-bold text-emerald-400 tracking-widest">SIMULATION ACTIVE</p><Info label="Incident ID" value={createdIncidentId}/><Info label="Scenario" value={startResponse?.scenarioId}/><Info label="Location" value={startResponse?.locationName}/><Info label="Coordinates" value={`${startResponse?.latitude}, ${startResponse?.longitude}`}/><Info label="Status" value={startResponse?.status||'RUNNING'}/></div>}
        <div className="flex gap-3"><button onClick={handleStartSimulation} disabled={!selectedScenarioId||!selectedCamera||simulationStatus==='STARTING'||simulationStatus==='RUNNING'} className="flex-1 min-h-[50px] bg-blue-600 rounded font-bold text-xs tracking-widest disabled:opacity-40">{simulationStatus==='STARTING'?'STARTING...':simulationStatus==='RUNNING'?'SIMULATION RUNNING':'START SIMULATION'}</button>{simulationStatus!=='IDLE'&&<button onClick={reset} className="border border-slate-600 rounded px-4 text-xs font-bold">RESET SIMULATION</button>}</div>
      </div></section>
      <section className="bg-slate-900 border border-slate-700 rounded-xl overflow-hidden"><div className="p-5 border-b border-slate-700 bg-slate-800/30 flex justify-between gap-3"><div><p className="text-[10px] font-bold text-teal-400 tracking-widest">STEP 2</p><h2 className="mt-1 font-bold uppercase">Hospital Emergency Response</h2><p className="mt-1 text-xs text-slate-400">The incident created above is selected automatically.</p></div><button onClick={()=>refreshActiveIncidents().catch(error=>setHospitalError(error.message))} className="self-start border border-slate-600 rounded px-3 py-2 text-[10px] font-bold">REFRESH</button></div><div className="p-5 space-y-6">
        <div><label className="block mb-2 text-[10px] font-bold uppercase tracking-widest text-slate-500">Active Database Incidents</label><select value={selectedIncidentId} onChange={e=>setSelectedIncidentId(e.target.value)} disabled={hospitalLoading} className="w-full bg-slate-950 border border-slate-700 rounded px-3 py-3 text-sm"><option value="">-- Select Active Incident --</option>{activeIncidents.map(incident=><option key={incident.incident_id} value={incident.incident_id}>{show(incident.severity)} — {show(incident.incident_type)} — {show(incident.location)}</option>)}</select></div>
        <div className="bg-slate-950 border border-slate-800 rounded p-4"><p className="mb-3 pb-2 border-b border-slate-800 text-[10px] font-bold uppercase tracking-widest text-slate-500">Active Incident</p>{selectedIncident?<div className="grid grid-cols-2 gap-3 text-xs"><Info label="Incident ID" value={selectedIncident.incident_id}/><Info label="Severity" value={selectedIncident.severity}/><Info label="Type" value={selectedIncident.incident_type}/><Info label="Location" value={selectedIncident.location}/><Info label="Latitude" value={selectedIncident.latitude}/><Info label="Longitude" value={selectedIncident.longitude}/><Info label="Patients" value={selectedIncident.patient_count}/><Info label="Critical Patients" value={selectedIncident.critical_patient_count}/><Info label="Serious Patients" value={selectedIncident.serious_patient_count}/><Info label="Ambulances Required" value={selectedIncident.ambulances_required}/></div>:<p className="text-xs text-slate-500">Start a simulation to automatically select its incident.</p>}</div>
        <div className="bg-slate-950 border border-slate-800 rounded p-4 grid grid-cols-2 gap-3"><Info label="Hospital Response" value={simulationStatus==='RUNNING'?'READY / ACTIVE':'STANDBY'}/><Info label="Receiving Hospital" value={selectedIncident?.receiving_hospital_name||selectedIncident?.receiving_hospital_id}/><Info label="Hospital Agent" value={hospitalLoading?'EVALUATING…':hospitalSuccess?'ACTIVE':hospitalError?'ERROR':'AWAITING INCIDENT'}/><Info label="Traffic Coordination" value={simulationStatus==='RUNNING'?'READY':'STANDBY'}/></div>
        {hospitalError&&<Notice color="red" text={hospitalError}/>} {hospitalSuccess&&<Notice color="teal" text="HOSPITAL AGENT ACTIVE — assessment available for the selected incident."/>}
        <p className={`text-xs font-bold tracking-widest ${statusColor}`}>SIMULATION STATUS: {simulationStatus}</p>
      </div></section>
    </div>
  </main></div>;
}
function Info({label,value}) { return <div><p className="text-[10px] uppercase tracking-wider text-slate-500">{label}</p><p className="mt-1 font-mono text-slate-200 break-all">{show(value)}</p></div>; }
function Notice({color,text}) { const palette={red:'border-red-900/50 bg-red-950/30 text-red-300',blue:'border-blue-900/50 bg-blue-950/30 text-blue-300',teal:'border-teal-900/50 bg-teal-950/30 text-teal-300'}; return <div className={`border rounded p-3 text-xs ${palette[color]}`}>{text}</div>; }
