import React from 'react';

export default function SignalMonitoringTable({ signals, selectedSignalId, onSelect, isLoading }) {
  if (isLoading) {
    return <div className="p-8 text-center text-slate-500 animate-pulse font-mono">Loading signal network...</div>;
  }

  if (!signals || signals.length === 0) {
    return <div className="p-8 text-center text-slate-500 italic">No signals match the current operational filters.</div>;
  }

  const getTrafficBadge = (level) => {
    const base = "px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider rounded border";
    switch(level?.toUpperCase()) {
      case 'CRITICAL': return `${base} bg-red-900/50 text-red-400 border-red-500`;
      case 'SEVERE': return `${base} bg-orange-900/50 text-orange-400 border-orange-500`;
      case 'HIGH': return `${base} bg-yellow-900/50 text-yellow-400 border-yellow-500`;
      case 'LOW': return `${base} bg-emerald-900/50 text-emerald-400 border-emerald-500`;
      default: return `${base} bg-slate-800 text-slate-300 border-slate-600`; // MODERATE
    }
  };

  return (
    <div className="overflow-x-auto custom-scrollbar w-full h-full bg-slate-950">
      <table className="w-full table-fixed text-left text-sm whitespace-nowrap min-w-[1450px]">
        <thead className="bg-slate-900 text-[10px] uppercase tracking-widest text-slate-400 sticky top-0 z-10 border-b border-slate-800 shadow-sm">
          <tr>
            <th className="w-[130px] px-4 py-3 font-bold">Signal</th>
            <th className="w-[180px] px-4 py-3 font-bold">Intersection</th>
            <th className="w-[120px] px-4 py-3 font-bold">Area</th>
            <th className="w-[100px] px-4 py-3 font-bold">Traffic</th>
            <th className="w-[100px] px-4 py-3 font-bold">Density</th>
            <th className="w-[90px] px-4 py-3 font-bold">Phase</th>
            <th className="w-[70px] px-4 py-3 font-bold">Green</th>
            <th className="w-[70px] px-4 py-3 font-bold">Red</th>
            <th className="w-[180px] px-4 py-3 font-bold">AI Decision</th>
            <th className="w-[180px] px-4 py-3 font-bold">Recommended Change</th>
            <th className="w-[120px] px-4 py-3 font-bold">Forecast</th>
            <th className="w-[200px] px-4 py-3 font-bold">Impact</th>
            <th className="w-[150px] px-4 py-3 font-bold">Status</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/50">
          {signals.map((s) => {
            const isSelected = selectedSignalId === (s.signalId || s.signal_code);
            const isHighPriority = ['CRITICAL', 'SEVERE'].includes(s.trafficLevel || s.traffic_level);
            
            return (
              <tr 
                key={s.signalId || s.signal_code} 
                onClick={() => onSelect(s)}
                className={`cursor-pointer transition-colors ${isSelected ? 'bg-blue-900/20' : 'hover:bg-slate-800/50'}`}
              >
                <td className={`px-4 py-3 font-mono text-xs border-l-2 ${isSelected ? 'border-blue-500 text-blue-400' : isHighPriority ? 'border-red-500/50 text-red-300' : 'border-transparent text-slate-400'}`}>
                  {s.signalId || s.signal_code}
                </td>
                <td className="px-4 py-3 font-bold text-slate-200 truncate" title={s.intersection || s.intersection_name}>{s.intersection || s.intersection_name}</td>
                <td className="px-4 py-3 text-slate-400 truncate" title={s.area}>{s.area || 'Mumbai'}</td>
                <td className="px-4 py-3">
                  <span className={getTrafficBadge(s.trafficLevel || s.traffic_level)}>
                    {s.trafficLevel || s.traffic_level || 'MODERATE'}
                  </span>
                </td>
                <td className="px-4 py-3 text-slate-300 truncate">{s.densityLevel || s.density_level || 'MODERATE'}</td>
                <td className="px-4 py-3 text-slate-300">{s.currentPhase || s.current_phase || 'GREEN'}</td>
                <td className="px-4 py-3 font-mono text-emerald-400">{s.currentGreen || s.current_green_seconds || '--'}s</td>
                <td className="px-4 py-3 font-mono text-red-400">{s.currentRed || s.current_red_seconds || '--'}s</td>
                
                <td className="px-4 py-3 text-slate-300 truncate" title={s.aiDecision || s.ai_decision || 'Monitoring'}>
                  {s.aiDecision || s.ai_decision || 'Monitoring'}
                </td>
                <td className="px-4 py-3 font-bold text-emerald-400 truncate" title={s.recommendation || s.ai_recommendation || 'No change'}>
                  {s.recommendation || s.ai_recommendation || 'No change'}
                </td>
                
                <td className="px-4 py-3 text-orange-400 text-xs font-bold uppercase truncate">
                  {s.prediction || s.predicted_traffic_level || 'Stable'}
                </td>
                <td className="px-4 py-3 text-slate-400 text-xs truncate" title={s.expectedImpact || s.expected_impact || 'Maintain intersection throughput'}>
                  {s.expectedImpact || s.expected_impact || 'Maintain intersection throughput'}
                </td>
                <td className="px-4 py-3 text-xs tracking-wider uppercase font-bold text-blue-400 truncate">
                  {(s.status === 'SIMULATED' || s.status?.includes('DEMO')) ? 'RECOMMENDATION READY' : (s.status || 'MONITORING')}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}