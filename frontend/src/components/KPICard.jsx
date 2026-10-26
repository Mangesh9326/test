import React from 'react';

export default function KPICard({ title, value, status = 'neutral', icon }) {
  const statusColors = {
    critical: 'text-red-400 border-red-500/30 bg-red-500/10',
    warning: 'text-orange-400 border-orange-500/30 bg-orange-500/10',
    nominal: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10',
    neutral: 'text-blue-400 border-blue-500/30 bg-blue-500/10'
  };

  return (
    <div className={`glass-panel p-4 flex flex-col justify-between border-l-4 ${statusColors[status].replace('text-', 'border-l-')}`}>
      <div className="flex justify-between items-start mb-2">
        <span className="text-xs font-bold text-gray-400 uppercase tracking-wider">{title}</span>
        {icon && <span className="text-gray-500">{icon}</span>}
      </div>
      <div className={`text-3xl font-bold ${statusColors[status].split(' ')[0]}`}>
        {value}
      </div>
    </div>
  );
}