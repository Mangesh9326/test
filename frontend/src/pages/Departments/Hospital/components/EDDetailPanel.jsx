import DetailBar from "./DetailBar";

export function EDDetailPanel({ hospital, isSimulation }) {
  if (!hospital) {
    return <div className="p-10 text-center text-slate-500 italic text-xs border border-dashed border-slate-700 rounded-xl bg-slate-900/30">Select a facility from the table to inspect its emergency profile.</div>;
  }

  const getPct = (occ, cap) => cap ? Math.round((occ / cap) * 100) : null;
  const pct = getPct(hospital.emergency_occupancy, hospital.emergency_capacity);

  return (
    <div className="flex flex-col gap-6">
      <div className="border-b border-slate-800/80 pb-4 shrink-0">
        <h3 className="font-bold text-xl text-white uppercase tracking-tight">{hospital.name}</h3>
        <div className="mt-2 flex items-center gap-2">
          <span className={`inline-flex px-2.5 py-1 rounded border text-[9px] font-bold tracking-widest uppercase ${hospital.capacity_level === 'CRITICAL' ? 'bg-red-500/10 border-red-500/30 text-red-400' : hospital.capacity_level === 'HIGH' ? 'bg-orange-500/10 border-orange-500/30 text-orange-400' : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'}`}>
            {hospital.capacity_level || 'NORMAL'} PRESSURE
          </span>
          <span className="text-[10px] text-slate-500 font-mono">ID: {hospital.hospital_id ? String(hospital.hospital_id).split('-')[0] : 'N/A'}</span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-y-5 gap-x-4 text-xs">
        <div className="flex flex-col bg-slate-950 p-3 rounded-lg border border-slate-800">
          <span className="text-[10px] text-slate-500 uppercase tracking-widest mb-1 font-bold">ER Capacity</span>
          <span className="text-lg font-mono text-slate-200">{hospital.emergency_capacity ?? 'N/A'}</span>
        </div>
        <div className="flex flex-col bg-slate-950 p-3 rounded-lg border border-slate-800">
          <span className="text-[10px] text-slate-500 uppercase tracking-widest mb-1 font-bold">Available ER Bays</span>
          <span className={`text-lg font-mono font-bold ${pct !== null && pct >= 90 ? 'text-red-400' : 'text-emerald-400'}`}>{hospital.emergency_capacity != null && hospital.emergency_occupancy != null ? Math.max(0, hospital.emergency_capacity - hospital.emergency_occupancy) : 'N/A'}</span>
        </div>
        <div className="flex flex-col bg-slate-950 p-3 rounded-lg border border-slate-800">
          <span className="text-[10px] text-slate-500 uppercase tracking-widest mb-1 font-bold">Occupied</span>
          <span className="text-lg font-mono text-slate-200">{hospital.emergency_occupancy ?? 'N/A'}</span>
        </div>
        
        <div className="col-span-2">
          <DetailBar label="Emergency Department Saturation" occupied={hospital.emergency_occupancy} total={hospital.emergency_capacity} colorClass={pct >= 90 ? 'bg-red-500' : pct >= 75 ? 'bg-orange-500' : 'bg-emerald-500'} />
        </div>

        <div className="flex flex-col bg-slate-950 p-3 rounded-lg border border-slate-800">
          <span className="text-[10px] text-slate-500 uppercase tracking-widest mb-1 font-bold">Inbound Queue</span>
          <span className="text-lg font-mono text-orange-400 font-bold">{hospital.ambulance_queue ?? 'N/A'}</span>
        </div>
        <div className="flex flex-col bg-slate-950 p-3 rounded-lg border border-slate-800">
          <span className="text-[10px] text-slate-500 uppercase tracking-widest mb-1 font-bold">Average Wait</span>
          <span className="text-lg font-mono text-orange-400 font-bold">{hospital.average_wait_minutes != null ? `${hospital.average_wait_minutes}m` : 'N/A'}</span>
        </div>
        
        <div className="flex flex-col col-span-2">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest mb-1">Telemetry Timestamp</span>
          <span className="font-mono text-slate-400">{hospital.last_observed_at ? new Date(hospital.last_observed_at).toLocaleString() : 'DATA UNAVAILABLE'}</span>
        </div>
      </div>

      {isSimulation && hospital.sim_impact && (
        <div className="mt-2 bg-indigo-950/20 border border-indigo-900/40 p-4 rounded-xl flex flex-col gap-3 shrink-0">
          <span className="text-[10px] font-bold uppercase tracking-widest text-indigo-400">Simulation Impact vs Baseline</span>
          <div className="grid grid-cols-3 text-[10px] font-mono text-slate-300 gap-2 text-center">
             <div className="bg-slate-900 p-2 rounded-lg border border-slate-800">
               <span className="block text-slate-500 mb-1">ER Occ</span>
               Base: {hospital.baseline_occupancy}<br/>
               <span className="text-indigo-400 font-bold">Now: {hospital.emergency_occupancy}</span><br/>
               <span className="text-xs text-red-400 font-bold mt-1 block">+{hospital.sim_impact.patients}</span>
             </div>
             <div className="bg-slate-900 p-2 rounded-lg border border-slate-800">
               <span className="block text-slate-500 mb-1">Queue</span>
               Base: {hospital.baseline_queue}<br/>
               <span className="text-orange-400 font-bold">Now: {hospital.ambulance_queue}</span>
             </div>
             <div className="bg-slate-900 p-2 rounded-lg border border-slate-800">
               <span className="block text-slate-500 mb-1">Wait</span>
               Base: {hospital.baseline_wait}m<br/>
               <span className="text-red-400 font-bold">Now: {hospital.average_wait_minutes}m</span><br/>
               <span className="text-xs text-red-400 font-bold mt-1 block">+{hospital.sim_impact.wait}m</span>
             </div>
          </div>
        </div>
      )}
    </div>
  );
}
