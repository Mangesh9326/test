import React, { useState, useMemo } from 'react';

export default function HospitalTable({ data, selectedId, onSelect }) {
  const [sortConfig, setSortConfig] = useState({ key: 'capacity_level', direction: 'desc' });

  const requestSort = (key) => {
    let direction = 'asc';
    if (sortConfig && sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  const sortedData = useMemo(() => {
    if (!data) return [];
    let sortableItems = [...data];
    
    if (sortConfig !== null) {
      sortableItems.sort((a, b) => {
        let aValue = a[sortConfig.key];
        let bValue = b[sortConfig.key];
        
        if (sortConfig.key === 'er_occupancy') {
           aValue = a.emergency_capacity ? (a.emergency_occupancy || 0) / a.emergency_capacity : 0;
           bValue = b.emergency_capacity ? (b.emergency_occupancy || 0) / b.emergency_capacity : 0;
        } else if (sortConfig.key === 'capacity_level' || sortConfig.key === 'emergency_level') {
           const weights = { CRITICAL: 4, HIGH: 3, MODERATE: 2, NORMAL: 1, STANDBY: 1 };
           aValue = weights[a[sortConfig.key]] || 0;
           bValue = weights[b[sortConfig.key]] || 0;
        } else if (sortConfig.key === 'patient_inflow') {
           const weights = { CRITICAL: 4, HIGH: 3, INCREASING: 2, NORMAL: 1 };
           aValue = weights[a[sortConfig.key] || 'NORMAL'] || 0;
           bValue = weights[b[sortConfig.key] || 'NORMAL'] || 0;
        } else if (sortConfig.key === 'average_wait_minutes') {
           aValue = aValue || 0;
           bValue = bValue || 0;
        }

        if (aValue < bValue) return sortConfig.direction === 'asc' ? -1 : 1;
        if (aValue > bValue) return sortConfig.direction === 'asc' ? 1 : -1;
        return 0;
      });
    }
    return sortableItems;
  }, [data, sortConfig]);

  if (!data || data.length === 0) {
    return (
      <div className="flex h-full flex-1 items-center justify-center flex-col gap-3 text-slate-500 font-mono text-sm py-16 italic border border-slate-800 rounded-lg bg-slate-900/50">
        <svg className="w-10 h-10 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <span>No hospitals match the current filter combination.</span>
      </div>
    );
  }

  const renderOccupancyBar = (occupied, total) => {
    if (total == null || total === 0) return <span className="text-slate-600 text-[10px] font-mono" title="Data Unavailable">DATA UNAVAILABLE</span>;
    const percent = Math.min(((occupied || 0) / total) * 100, 100);
    const colorClass = percent >= 90 ? 'bg-red-500' : percent >= 75 ? 'bg-orange-500' : 'bg-emerald-500';
    return (
      <div className="flex flex-col gap-1 w-full mt-1">
        <div className="flex justify-between items-center text-[9px] text-slate-400 font-mono leading-none">
           <span>{occupied || 0}/{total}</span>
           <span className={percent >= 90 ? 'text-red-400 font-bold' : ''}>{Math.round(percent)}%</span>
        </div>
        <div className="flex h-1.5 w-full rounded-full overflow-hidden bg-slate-800 border border-slate-700">
          <div className={`transition-all duration-500 ${colorClass}`} style={{ width: `${percent}%` }}></div>
        </div>
      </div>
    );
  };

  const SortIcon = ({ columnKey }) => {
    if (sortConfig?.key !== columnKey) return <span className="text-slate-700 ml-1 font-mono">↕</span>;
    return <span className="text-blue-400 ml-1 font-mono">{sortConfig.direction === 'asc' ? '▲' : '▼'}</span>;
  };

  const SortableTH = ({ label, columnKey, width, align = 'left', tooltip }) => (
    <th 
      className={`${width} px-4 py-3 font-bold cursor-pointer hover:bg-slate-800 transition-colors text-${align} group`}
      onClick={() => requestSort(columnKey)}
      title={tooltip}
      tabIndex="0"
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); requestSort(columnKey); }}}
    >
      <div className={`flex items-center ${align === 'right' ? 'justify-end' : ''}`}>
        <span className="group-hover:text-blue-300 transition-colors underline decoration-slate-600 decoration-dashed underline-offset-4">{label}</span>
        <SortIcon columnKey={columnKey} />
      </div>
    </th>
  );

  return (
    <div className="overflow-x-auto custom-scrollbar w-full h-full bg-slate-900 flex-1 relative">
      <table className="w-full table-fixed text-left text-sm whitespace-nowrap min-w-[1400px]">
        <thead className="bg-slate-950/90 text-[9px] uppercase tracking-widest text-slate-500 sticky top-0 z-10 border-b border-slate-800 shadow-sm backdrop-blur-md">
          <tr>
            <SortableTH label="Hospital" columnKey="name" width="w-[200px]" />
            <SortableTH label="Area" columnKey="area" width="w-[120px]" />
            <th className="w-[140px] px-4 py-3 font-bold">Type</th>
            <SortableTH label="Emergency" columnKey="emergency_level" width="w-[130px]" tooltip="Emergency readiness status" />
            <SortableTH label="Beds" columnKey="available_beds" width="w-[120px]" />
            <SortableTH label="ICU" columnKey="icu_available" width="w-[120px]" tooltip="Available intensive-care capacity. (Operational metric)" />
            <SortableTH label="ER Load" columnKey="er_occupancy" width="w-[120px]" tooltip="Current emergency department occupancy relative to configured capacity. (Operational metric)" />
            <SortableTH label="Wait" columnKey="average_wait_minutes" width="w-[90px]" align="right" />
            <SortableTH label="Inflow" columnKey="patient_inflow" width="w-[100px]" tooltip="Current operational classification of incoming patient volume. (Operational metric)" />
            <SortableTH label="Status" columnKey="capacity_level" width="w-[110px]" tooltip="Application-defined operational pressure level. (Operational metric)" />
            <th className="w-[140px] px-4 py-3 font-bold">Agent Status</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/50">
          {sortedData.map((h) => {
            const isSelected = selectedId === h.hospital_id;
            const isCritical = h.capacity_level === 'CRITICAL' || h.capacity_level === 'HIGH';
            
            let emerStatus = "ACCEPTING";
            let emerDot = "bg-emerald-500";
            if (!h.emergency_available) { emerStatus = "UNAVAILABLE"; emerDot = "bg-slate-600"; }
            else if (h.emergency_level === 'CRITICAL' || h.emergency_level === 'HIGH') { emerStatus = "RESTRICTED"; emerDot = "bg-orange-500"; }

            let inflowColor = "text-emerald-400";
            const inflowVal = h.patient_inflow || 'NORMAL';
            if (inflowVal === 'CRITICAL') inflowColor = "text-red-500";
            else if (inflowVal === 'HIGH') inflowColor = "text-red-400";
            else if (inflowVal === 'INCREASING') inflowColor = "text-orange-400";

            let capColor = "text-emerald-400";
            const capVal = h.capacity_level || 'NORMAL';
            if (capVal === 'CRITICAL') capColor = "text-red-500";
            else if (capVal === 'HIGH') capColor = "text-orange-400";
            else if (capVal === 'MODERATE') capColor = "text-yellow-400";

            const handleRowSelect = () => isSelected ? onSelect(null) : onSelect(h.hospital_id);

            return (
              <tr 
                key={h.hospital_id} 
                onClick={handleRowSelect}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleRowSelect(); }}}
                className={`cursor-pointer transition-colors focus-visible:outline-none focus-visible:bg-slate-800 ${isSelected ? 'bg-blue-900/20 hover:bg-blue-900/30' : 'hover:bg-slate-800/50'}`}
                tabIndex="0"
                aria-selected={isSelected}
              >
                <td className={`px-4 py-3 font-bold text-slate-200 truncate border-l-2 ${isSelected ? 'border-blue-500 text-blue-400' : isCritical ? 'border-red-500' : 'border-transparent'}`} title={h.name}>
                  {h.name}
                </td>
                <td className="px-4 py-3 text-slate-400 text-xs truncate" title={h.area}>{h.area}</td>
                <td className="px-4 py-3 text-slate-500 text-[10px] uppercase tracking-wider truncate" title={h.hospital_type}>{h.hospital_type}</td>
                
                <td className="px-4 py-3">
                  <div className="flex items-center gap-1.5">
                    <span className={`w-2 h-2 rounded-full ${emerDot} shadow-sm`}></span>
                    <span className={`text-[9px] font-bold uppercase tracking-wider ${!h.emergency_available ? 'text-slate-500' : 'text-slate-300'}`}>
                      {emerStatus}
                    </span>
                  </div>
                </td>

                <td className="px-4 py-3">{renderOccupancyBar(h.occupied_beds, h.total_beds)}</td>
                <td className="px-4 py-3">{renderOccupancyBar(h.icu_occupied, h.icu_total)}</td>
                <td className="px-4 py-3">{renderOccupancyBar(h.emergency_occupancy, h.emergency_capacity)}</td>
                
                <td className="px-4 py-3 font-mono text-xs text-right">
                   {h.average_wait_minutes != null ? (
                      <span className={h.average_wait_minutes > 30 ? 'text-orange-400 font-bold' : 'text-slate-300'}>{h.average_wait_minutes}m</span>
                   ) : <span className="text-slate-600">N/A</span>}
                </td>

                <td className="px-4 py-3">
                  <span className={`text-[9px] font-bold uppercase tracking-wider ${inflowColor}`}>{inflowVal}</span>
                </td>

                <td className="px-4 py-3">
                  <span className={`text-[10px] font-bold uppercase tracking-wider ${capColor}`}>{capVal}</span>
                </td>

                <td className="px-4 py-3">
                  <span className={`text-[9px] font-bold uppercase tracking-widest px-2 py-0.5 rounded border ${isCritical ? 'bg-red-900/20 text-red-400 border-red-800/50' : 'bg-blue-900/20 text-blue-400 border-blue-800/50'}`}>
                    {isCritical ? 'DIVERTING' : 'MONITORING'}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}