import React from 'react';

const TRAFFIC_STYLES = {
  CRITICAL: 'bg-red-900/50 text-red-400 border-red-500',
  SEVERE: 'bg-orange-900/50 text-orange-400 border-orange-500',
  HIGH: 'bg-yellow-900/50 text-yellow-400 border-yellow-500',
};
const DEFAULT_TRAFFIC_STYLE = 'bg-emerald-900/50 text-emerald-400 border-emerald-500';

// Left accent bar per severity, so the table can be scanned at a glance
const ACCENT_BORDER = {
  CRITICAL: 'border-red-500',
  SEVERE: 'border-orange-500',
  HIGH: 'border-yellow-500',
};
const DEFAULT_ACCENT_BORDER = 'border-emerald-600';

export default function SignalMonitoringTable({ signals, selectedSignalId, onSelect }) {
  if (!signals || signals.length === 0) {
    return <div className="text-center text-slate-500 py-10 italic px-4">No signals match the current criteria.</div>;
  }

  const getTrafficColor = (level) => TRAFFIC_STYLES[level] || DEFAULT_TRAFFIC_STYLE;

  const handleRowKeyDown = (e, s) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onSelect(s);
    }
  };

  return (
    <div className="overflow-auto custom-scrollbar w-full h-full">
      <table className="w-full table-fixed text-left text-sm whitespace-nowrap min-w-[1400px]">
        <thead className="bg-slate-900 text-[10px] uppercase tracking-wide text-slate-400 sticky top-0 z-20 border-b border-slate-700">
          <tr>
            <th className="w-[110px] px-4 py-3 sticky left-0 z-30 bg-slate-900 shadow-[2px_0_0_0_rgba(51,65,85,1)]">Signal ID</th>
            <th className="w-[200px] px-4 py-3">Intersection</th>
            <th className="w-[120px] px-4 py-3">Area</th>
            <th className="w-[100px] px-4 py-3">Traffic</th>
            <th className="w-[100px] px-4 py-3">Density</th>
            <th className="w-[100px] px-4 py-3">Phase</th>
            <th className="w-[80px] px-4 py-3">Green</th>
            <th className="w-[80px] px-4 py-3">Red</th>
            <th className="w-[180px] px-4 py-3">AI decision</th>
            <th className="w-[150px] px-4 py-3">Recommendation</th>
            <th className="w-[120px] px-4 py-3">Forecast</th>
            <th className="w-[160px] px-4 py-3">Status</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {signals.map((s, idx) => {
            const isSelected = selectedSignalId === s.signal_code;
            const accent = isSelected ? 'border-blue-500' : (ACCENT_BORDER[s.traffic_level] || DEFAULT_ACCENT_BORDER);
            const rowBg = isSelected ? 'bg-blue-900/20' : idx % 2 === 1 ? 'bg-slate-900/40' : '';
            const cellBg = isSelected ? 'bg-blue-950' : idx % 2 === 1 ? 'bg-slate-900' : 'bg-slate-950';
            return (
              <tr
                key={s.signal_code}
                onClick={() => onSelect(s)}
                onKeyDown={(e) => handleRowKeyDown(e, s)}
                tabIndex={0}
                role="button"
                aria-pressed={isSelected}
                className={`cursor-pointer transition-colors outline-none focus-visible:bg-slate-800/70 hover:bg-slate-800/50 ${rowBg}`}
              >
                <td className={`px-4 py-3 font-mono text-xs sticky left-0 z-10 border-l-2 shadow-[2px_0_0_0_rgba(30,41,59,1)] ${accent} ${cellBg} ${isSelected ? 'text-blue-400' : 'text-slate-400'}`}>
                  {s.signal_code}
                </td>
                <td className="px-4 py-3 font-semibold text-slate-200 truncate" title={s.intersection_name}>{s.intersection_name}</td>
                <td className="px-4 py-3 text-slate-400 truncate">{s.area}</td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide rounded border ${getTrafficColor(s.traffic_level)}`}>
                    {s.traffic_level}
                  </span>
                </td>
                <td className="px-4 py-3 text-slate-300">{s.density_level}</td>
                <td className="px-4 py-3 text-slate-300">{s.current_phase}</td>
                <td className="px-4 py-3 font-mono text-emerald-400">{s.current_green_seconds}s</td>
                <td className="px-4 py-3 font-mono text-red-400">{s.current_red_seconds}s</td>
                <td className="px-4 py-3 text-slate-300 truncate" title={s.ai_decision}>{s.ai_decision}</td>
                <td className="px-4 py-3 font-bold text-emerald-400 truncate" title={s.ai_recommendation}>{s.ai_recommendation}</td>
                <td className="px-4 py-3 text-orange-400 text-xs font-bold uppercase">{s.predicted_traffic_level || 'N/A'}</td>
                <td className="px-4 py-3 text-xs tracking-wide uppercase font-bold text-blue-400 truncate" title={s.status}>{s.status}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {/* Custom scrollbar styling — shared by any element using the "custom-scrollbar" class */}
      <style>{`
        .custom-scrollbar {
          scrollbar-width: thin;
          scrollbar-color: #334155 transparent;
        }
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
          height: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background-color: #334155;
          border-radius: 9999px;
          border: 2px solid transparent;
          background-clip: padding-box;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background-color: #475569;
        }
        .custom-scrollbar::-webkit-scrollbar-corner {
          background: transparent;
        }
      `}</style>
    </div>
  );
}