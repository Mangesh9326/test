import React from "react";

const Bar = ({ className = "" }) => <div className={`bg-slate-800 rounded animate-pulse ${className}`} />;

export const KpiSkeleton = () => (
  <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-6 gap-3">
    {Array.from({ length: 6 }).map((_, i) => (
      <div key={i} className="p-3 rounded-lg border border-slate-800 bg-slate-900">
        <Bar className="h-2.5 w-16 mb-3" />
        <Bar className="h-7 w-10" />
      </div>
    ))}
  </div>
);

export const TableSkeleton = ({ rows = 8 }) => (
  <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 space-y-2">
    {Array.from({ length: rows }).map((_, i) => (
      <Bar key={i} className="h-10 w-full" />
    ))}
  </div>
);

export const DetailsSkeleton = () => (
  <div className="space-y-4">
    <Bar className="h-10 w-72" />
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
      <Bar className="lg:col-span-8 h-[420px]" />
      <div className="lg:col-span-4 space-y-4">
        <Bar className="h-44" /><Bar className="h-36" /><Bar className="h-44" />
      </div>
    </div>
  </div>
);

export const EmptyState = () => (
  <div className="bg-slate-900 border border-slate-700 rounded-lg py-16 text-center">
    <span className="inline-block w-2.5 h-2.5 rounded-full bg-emerald-500 mb-3" />
    <div className="text-sm font-bold uppercase tracking-widest text-slate-200">No incidents detected</div>
    <p className="text-xs text-slate-400 mt-1">Current police operations are clear.</p>
  </div>
);

export const NoMatches = ({ onClear }) => (
  <div className="bg-slate-900 border border-slate-700 rounded-lg py-12 text-center">
    <div className="text-sm font-bold uppercase tracking-widest text-slate-300">No matching incidents</div>
    <p className="text-xs text-slate-400 mt-1 mb-4">Try adjusting or clearing your filters.</p>
    <button type="button" onClick={onClear} className="text-[11px] font-bold uppercase tracking-wider px-3 py-1.5 rounded border border-blue-500/40 text-blue-300 bg-blue-500/10 hover:bg-blue-500/20">
      Clear Filters
    </button>
  </div>
);

export const ErrorState = ({ onRetry, title = "Unable to load incident data", message = "Police incident services are temporarily unavailable." }) => (
  <div className="bg-slate-900 border border-red-500/30 rounded-lg py-14 text-center">
    <span className="inline-block w-2.5 h-2.5 rounded-full bg-red-500 mb-3" />
    <div className="text-sm font-bold uppercase tracking-widest text-red-300">{title}</div>
    <p className="text-xs text-slate-400 mt-1 mb-4">{message}</p>
    <button type="button" onClick={onRetry} className="text-[11px] font-bold uppercase tracking-wider px-4 py-1.5 rounded border border-red-500/40 text-red-300 bg-red-500/10 hover:bg-red-500/20">
      Retry
    </button>
  </div>
);
