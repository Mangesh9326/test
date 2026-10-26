export function EDTable({ hospitals, selectedId, onSelect, isSimulation }) {
  if (!hospitals || hospitals.length === 0) {
    return <div className="p-8 text-center text-slate-500 italic text-sm border-t border-slate-800">DATA NOT AVAILABLE</div>;
  }

  const getBadgeColor = (level) => {
    switch (level) {
      case "CRITICAL": return "bg-red-500/10 border-red-500/30 text-red-400";
      case "HIGH": return "bg-orange-500/10 border-orange-500/30 text-orange-400";
      case "ELEVATED": return "bg-yellow-500/10 border-yellow-500/30 text-yellow-400";
      default: return "bg-emerald-500/10 border-emerald-500/30 text-emerald-400";
    }
  };

  return (
    <div className="overflow-x-auto w-full custom-scrollbar">
      <table className="w-full min-w-[880px] text-left text-xs whitespace-nowrap">
        <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-700 uppercase tracking-widest text-[10px]">
          <tr>
            <th className="p-4 font-bold">Hospital Facility</th>
            <th className="p-4 font-bold">ER Level</th>
            <th className="p-4 font-bold text-center">Inbound Queue</th>
            <th className="p-4 font-bold text-right">Avg Wait</th>
            <th className="p-4 font-bold text-center">Occupancy</th>
            <th className="p-4 font-bold text-center">Available</th>
            <th className="p-4 font-bold text-right">Pressure</th>
            {isSimulation && <th className="p-4 font-bold text-center">Trend</th>}
            <th className="p-4 font-bold">Operational Status</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/60 bg-slate-950/30">
          {hospitals.map((h) => {
            const isSelected = selectedId === h.hospital_id;
            const capLevel = h.capacity_level || "NORMAL";
            const pressPct = h.emergency_capacity ? Math.round((h.emergency_occupancy / h.emergency_capacity) * 100) : null;
            const available = h.emergency_capacity != null && h.emergency_occupancy != null ? Math.max(0, h.emergency_capacity - h.emergency_occupancy) : null;

            return (
              <tr 
                key={h.hospital_id} 
                onClick={() => onSelect(h.hospital_id)}
                className={`cursor-pointer transition-colors outline-none focus-visible:bg-slate-800/80 ${isSelected ? "bg-blue-900/20" : "hover:bg-slate-800/60"}`}
                tabIndex="0"
                onKeyDown={(e) => { if(e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(h.hospital_id); }}}
              >
                <td className={`p-4 font-bold text-slate-200 border-l-[3px] ${isSelected ? 'border-blue-500' : 'border-transparent'}`}>{h.name}</td>
                <td className="p-4">
                  <span className={`px-2.5 py-1 rounded border text-[9px] font-bold tracking-widest ${getBadgeColor(h.emergency_level || capLevel)}`}>
                    {h.emergency_level || capLevel}
                  </span>
                </td>
                <td className="p-4 font-mono text-slate-300 text-center">{h.ambulance_queue ?? 'N/A'}</td>
                <td className="p-4 font-mono text-orange-400 text-right">{h.average_wait_minutes != null ? `${h.average_wait_minutes}m` : 'N/A'}</td>
                <td className="p-4 font-mono text-slate-300 text-center">{h.emergency_occupancy ?? 'N/A'} / {h.emergency_capacity ?? 'N/A'}</td>
                <td className={`p-4 font-mono text-center font-bold ${available !== null && available <= 3 ? 'text-red-400' : 'text-emerald-400'}`}>{available ?? 'N/A'}</td>
                <td className={`p-4 font-mono text-right font-bold ${pressPct >= 90 ? 'text-red-400' : 'text-slate-300'}`}>{pressPct != null ? `${pressPct}%` : 'N/A'}</td>
                {isSimulation && (
                  <td className="p-4 font-mono text-[10px] text-center">
                    <span className={(h.trend || '').includes('Increasing') ? 'text-red-400' : (h.trend || '').includes('Recovering') ? 'text-emerald-400' : 'text-slate-400'}>
                      {h.trend || "→ Stable"}
                    </span>
                  </td>
                )}
                <td className="p-4">
                  <span className="text-[10px] font-bold tracking-widest uppercase text-slate-400">{h.status === 'Active' ? 'OPERATIONAL' : 'OFFLINE'}</span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
