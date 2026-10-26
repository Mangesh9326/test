const tone = (value = '') => {
  const upper = String(value).toUpperCase();
  if (upper.includes('CRITICAL') || upper.includes('WORSEN')) return 'border-red-500/40 bg-red-500/10 text-red-300';
  if (upper.includes('HIGH') || upper.includes('ELEVATED') || upper.includes('REVIEW')) return 'border-amber-500/40 bg-amber-500/10 text-amber-300';
  if (upper.includes('IMPROV') || upper.includes('NORMAL') || upper.includes('STABLE')) return 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300';
  return 'border-sky-500/40 bg-sky-500/10 text-sky-300';
};

export function Badge({ children }) { return <span className={`inline-flex rounded border px-2 py-1 font-mono text-[10px] font-bold tracking-wider ${tone(children)}`}>{children || 'N/A'}</span>; }
export function AgentCard({ title, eyebrow, children, className = '' }) { return <section className={`rounded-xl border border-slate-800 bg-slate-900/70 shadow-xl shadow-slate-950/20 ${className}`}><div className="border-b border-slate-800 px-5 py-4">{eyebrow && <p className="font-mono text-[10px] font-bold tracking-[.18em] text-sky-400">{eyebrow}</p>}<h2 className="mt-1 text-sm font-bold tracking-wide text-white">{title}</h2></div><div className="p-5">{children}</div></section>; }
