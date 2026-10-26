import React from "react";

export default function Panel({ title, right, className = "", children }) {
  return (
    <section className={`bg-slate-900 border border-slate-700 rounded-lg shadow-lg flex flex-col min-h-0 ${className}`}>
      <div className="px-4 py-2.5 border-b border-slate-700 flex justify-between items-center gap-2 shrink-0">
        <h3 className="text-xs font-bold text-slate-300 uppercase tracking-widest">{title}</h3>
        {right}
      </div>
      {children}
    </section>
  );
}

export const Field = ({ label, children, mono = false }) => (
  <div className="py-1.5 border-b border-slate-800 last:border-0">
    <div className="text-[9px] uppercase tracking-widest text-slate-500">{label}</div>
    <div className={`text-xs text-slate-200 break-words ${mono ? "font-mono" : ""}`}>{children}</div>
  </div>
);
