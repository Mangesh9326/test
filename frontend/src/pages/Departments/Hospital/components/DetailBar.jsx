export default function DetailBar({ label, occupied, total, colorClass }) {
  if (total == null || total === 0) {
    return (
      <div className="flex flex-col gap-1 w-full mt-1" role="progressbar" aria-label={label} aria-valuetext="Data Unavailable">
        <div className="flex justify-between items-center text-[9px] text-slate-400 font-mono leading-none mb-0.5">
           <span className="uppercase tracking-widest">{label}</span>
           <span title="Data Unavailable">N/A</span>
        </div>
        <div className="h-1.5 w-full bg-slate-800/80 rounded-full overflow-hidden border border-slate-700/50"></div>
      </div>
    );
  }

  const pct = Math.min(((occupied || 0) / total) * 100, 100);
  
  return (
    <div className="flex flex-col gap-1 w-full mt-1" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin="0" aria-valuemax="100" aria-label={label}>
      <div className="flex justify-between items-center text-[9px] text-slate-400 font-mono leading-none mb-0.5">
         <span className="uppercase tracking-widest">{label}</span>
         <span>{Math.round(pct)}%</span>
      </div>
      <div className="h-1.5 w-full bg-slate-800/80 rounded-full overflow-hidden border border-slate-700/50">
         <div className={`h-full transition-all duration-500 ${colorClass || 'bg-blue-500'}`} style={{ width: `${pct}%` }}></div>
      </div>
    </div>
  );
}
