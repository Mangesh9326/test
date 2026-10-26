import React, { useState, useEffect, useMemo } from 'react';

export default function ResourceManagement() {
  const [resources, setResources] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  // Filters
  const [filterHospital, setFilterHospital] = useState('All');
  const [filterResource, setFilterResource] = useState('All');

  useEffect(() => {
    const fetchResources = async () => {
      try {
        setLoading(true);
        setError(false);
        const res = await fetch('/api/hospitals/resources');
        if (!res.ok) throw new Error('API Error');
        const data = await res.json();
        setResources(data);
      } catch (err) {
        console.error('[HOSPITAL UI] Failed to load resources:', err);
        setError(true);
      } finally {
        setLoading(false);
      }
    };

    fetchResources();
    const interval = setInterval(fetchResources, 30000);
    return () => clearInterval(interval);
  }, []);

  // Filter Dropdowns
  const uniqueHospitals = useMemo(() => ['All', ...new Set(resources.map(r => r.hospital_name).filter(Boolean))].sort(), [resources]);
  const uniqueResourceTypes = useMemo(() => ['All', ...new Set(resources.map(r => r.resource_type).filter(Boolean))].sort(), [resources]);

  // Derived Network-Level Aggregates
  const networkSummary = useMemo(() => {
    const summary = {};
    resources.forEach(r => {
      if (!summary[r.resource_type]) {
        summary[r.resource_type] = { total: 0, available: 0, inUse: 0 };
      }
      summary[r.resource_type].total += Number(r.total_capacity);
      summary[r.resource_type].available += Number(r.available_qty);
      summary[r.resource_type].inUse += Number(r.in_use_qty);
    });
    return summary;
  }, [resources]);

  // Active Alerts (HIGH or CRITICAL)
  const activeAlerts = useMemo(() => resources.filter(r => r.status === 'CRITICAL' || r.status === 'HIGH'), [resources]);

  // Filtered Table Data
  const filteredData = useMemo(() => {
    return resources.filter(r => {
      if (filterHospital !== 'All' && r.hospital_name !== filterHospital) return false;
      if (filterResource !== 'All' && r.resource_type !== filterResource) return false;
      return true;
    });
  }, [resources, filterHospital, filterResource]);

  const renderUtilizationBar = (inUse, total, threshold) => {
    if (!total) return null;
    const percent = Math.min((inUse / total) * 100, 100);
    const colorClass = percent >= threshold ? 'bg-red-500' : percent >= threshold - 15 ? 'bg-orange-500' : 'bg-emerald-500';
    return (
      <div className="flex flex-col gap-1 w-full mt-1">
        <div className="flex h-1.5 w-full rounded-full overflow-hidden bg-slate-800 border border-slate-700 relative">
          <div className={`transition-all duration-500 ${colorClass}`} style={{ width: `${percent}%` }}></div>
          <div className="absolute top-0 bottom-0 w-0.5 bg-yellow-400" style={{ left: `${threshold}%` }} title="Threshold"></div>
        </div>
        <span className="text-[9px] text-slate-500 font-mono text-right">{percent.toFixed(1)}%</span>
      </div>
    );
  };

  if (loading) return <div className="flex h-full items-center justify-center font-mono text-slate-400 animate-pulse">Synchronizing Network Resources...</div>;
  if (error) return <div className="p-6 text-red-400 font-bold uppercase tracking-widest text-sm">Error connecting to Hospital API</div>;

  return (
    <div className="flex flex-col gap-6 h-full p-6 text-slate-200 bg-slate-950 overflow-y-auto font-sans">
      
      <header className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4 shrink-0 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-3xl font-bold uppercase tracking-tight text-white">Resource Management</h1>
          <p className="text-sm text-slate-400 mt-1">Network Medical Infrastructure & Consumables[cite: 1]</p>
        </div>
        <div className="flex items-center gap-4">
          <select 
            value={filterHospital} 
            onChange={(e) => setFilterHospital(e.target.value)} 
            className="bg-slate-900 border border-slate-700 text-xs font-bold rounded px-4 py-2 outline-none text-slate-200"
          >
            <option value="All" disabled>Filter Hospital...</option>
            {uniqueHospitals.map(h => <option key={h} value={h}>{h}</option>)}
          </select>
          <select 
            value={filterResource} 
            onChange={(e) => setFilterResource(e.target.value)} 
            className="bg-slate-900 border border-slate-700 text-xs font-bold rounded px-4 py-2 outline-none text-slate-200"
          >
            <option value="All" disabled>Filter Resource...</option>
            {uniqueResourceTypes.map(r => <option key={r} value={r}>{r}</option>)}
          </select>
        </div>
      </header>

      {/* Network-Level Resource Summary */}
      <div className="shrink-0">
        <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3">Network Summary (Aggregated)</h2>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {Object.entries(networkSummary).map(([type, stats]) => (
            <div key={type} className="bg-slate-900 border border-slate-800 p-4 rounded-xl shadow-lg flex flex-col justify-between">
              <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-2">{type}</span>
              <div className="flex justify-between items-end">
                <div className="flex flex-col">
                  <span className="text-[9px] text-slate-500 uppercase">Available</span>
                  <span className={`text-xl font-mono font-bold ${stats.available < stats.total * 0.15 ? 'text-red-400' : 'text-emerald-400'}`}>
                    {stats.available}
                  </span>
                </div>
                <div className="text-[10px] text-slate-400 font-mono">/ {stats.total}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 flex-1">
        
        {/* Main Resource Inventory Table */}
        <div className="xl:col-span-8 bg-slate-900 border border-slate-800 rounded-xl shadow-lg flex flex-col overflow-hidden h-full">
          <div className="p-4 border-b border-slate-800 bg-slate-800/30 flex justify-between items-center shrink-0">
            <h3 className="font-bold text-sm uppercase tracking-widest text-slate-200">Hospital Resource Inventory</h3>
          </div>
          <div className="overflow-x-auto flex-1 custom-scrollbar">
            <table className="w-full text-left text-sm whitespace-nowrap min-w-[900px]">
              <thead className="bg-slate-950/80 text-[10px] uppercase tracking-widest text-slate-500 sticky top-0 border-b border-slate-800 z-10">
                <tr>
                  <th className="px-4 py-3">Hospital</th>
                  <th className="px-4 py-3">Resource</th>
                  <th className="px-4 py-3 text-right">Total</th>
                  <th className="px-4 py-3 text-right">Available</th>
                  <th className="px-4 py-3 text-right">In Use</th>
                  <th className="px-4 py-3 w-[150px]">Utilization</th>
                  <th className="px-4 py-3 text-right">Threshold</th>
                  <th className="px-4 py-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {filteredData.map((row, idx) => {
                  const utilPercent = (Number(row.in_use_qty) / Number(row.total_capacity)) * 100;
                  return (
                    <tr key={idx} className="hover:bg-slate-800/50 transition-colors">
                      <td className="px-4 py-3 font-bold text-slate-200 truncate max-w-[200px]" title={row.hospital_name}>{row.hospital_name}</td>
                      <td className="px-4 py-3 text-slate-400 text-xs font-bold uppercase tracking-wider">{row.resource_type}</td>
                      <td className="px-4 py-3 font-mono text-slate-300 text-right">{row.total_capacity}</td>
                      <td className="px-4 py-3 font-mono text-emerald-400 text-right">{row.available_qty}</td>
                      <td className="px-4 py-3 font-mono text-orange-400 text-right">{row.in_use_qty}</td>
                      <td className="px-4 py-3">{renderUtilizationBar(Number(row.in_use_qty), Number(row.total_capacity), Number(row.threshold_percent))}</td>
                      <td className="px-4 py-3 font-mono text-slate-500 text-right text-[10px]">{Number(row.threshold_percent).toFixed(1)}%</td>
                      <td className="px-4 py-3">
                        <span className={`px-2 py-1 text-[9px] tracking-wider uppercase font-bold rounded ${
                          row.status === 'CRITICAL' ? 'bg-red-900/50 text-red-400' :
                          row.status === 'HIGH' ? 'bg-orange-900/50 text-orange-400' :
                          row.status === 'MONITOR' ? 'bg-blue-900/50 text-blue-400' :
                          'bg-emerald-900/50 text-emerald-400'
                        }`}>
                          {row.status}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {filteredData.length === 0 && <div className="text-center text-slate-500 text-sm py-8 italic">No resources match the selected filters.</div>}
          </div>
        </div>

        {/* Active Resource Alerts */}
        <div className="xl:col-span-4 flex flex-col gap-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl shadow-lg flex flex-col h-full overflow-hidden">
            <div className="p-4 border-b border-slate-800 bg-slate-800/30 flex justify-between items-center shrink-0">
              <h3 className="font-bold text-sm uppercase tracking-widest flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
                Resource Alerts
              </h3>
              <span className="text-[10px] font-mono text-slate-500 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">{activeAlerts.length} Active</span>
            </div>
            
            <div className="flex-1 overflow-y-auto custom-scrollbar p-4 bg-black">
              {activeAlerts.length === 0 ? (
                <div className="text-slate-600 text-xs italic flex items-center justify-center h-full">All network resources are within operational thresholds.</div>
              ) : (
                <div className="space-y-3">
                  {activeAlerts.map((alert, idx) => (
                    <div key={idx} className="bg-slate-950 border border-slate-800 p-3 rounded-lg relative overflow-hidden">
                      <div className={`absolute left-0 top-0 bottom-0 w-1 ${alert.status === 'CRITICAL' ? 'bg-red-500' : 'bg-orange-500'}`}></div>
                      <div className="ml-2 flex justify-between items-start mb-1">
                        <span className="text-slate-200 font-bold text-sm truncate pr-2">{alert.hospital_name}</span>
                        <span className={`text-[9px] font-bold uppercase tracking-widest px-2 py-0.5 rounded ${alert.status === 'CRITICAL' ? 'bg-red-900/30 text-red-400' : 'bg-orange-900/30 text-orange-400'}`}>
                          {alert.status}
                        </span>
                      </div>
                      <div className="ml-2 mt-2">
                        <span className="text-xs text-slate-400 font-bold uppercase tracking-wider">{alert.resource_type} Depletion Risk</span>
                        <div className="flex justify-between items-center mt-1 text-[11px] font-mono">
                          <span className="text-slate-500">Available: <span className="text-red-400 font-bold">{alert.available_qty}</span> / {alert.total_capacity}</span>
                          <span className="text-slate-500">Threshold: {Number(alert.threshold_percent).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}