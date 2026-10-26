import React, { useState, useEffect, useMemo } from 'react';

export default function HospitalCapacity() {
  const [networkData, setNetworkData] = useState([]);
  const [selectedHospital, setSelectedHospital] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  // Filters
  const [searchQuery, setSearchQuery] = useState('');
  const [filterArea, setFilterArea] = useState('All');
  const [filterRisk, setFilterRisk] = useState('All');

  useEffect(() => {
    const fetchCapacityData = async () => {
      try {
        setLoading(true);
        setError(false);
        
        // Fetch base capacity metrics and full monitoring context for demand/prediction data
        const [capacityRes, hospitalsRes, monitoringRes] = await Promise.all([
          fetch('/api/hospitals/capacity'),
          fetch('/api/hospitals?limit=100'),
          fetch('/api/hospitals/monitoring')
        ]);

        if (!capacityRes.ok || !hospitalsRes.ok || !monitoringRes.ok) throw new Error('API Error');

        const hospitals = (await hospitalsRes.json()).data;
        const monitoring = await monitoringRes.json();
        const capacity = await capacityRes.json();

        // Merge backend data to provide a unified capacity vs. demand view
        const merged = hospitals.map(h => {
          const mon = monitoring.find(m => m.hospital_id === h.hospital_id) || {};
          const cap = capacity.find(c => c.name === h.name) || {};
          
          return {
            ...h,
            ...mon,
            ...cap,
            // Expected backend fields mapped defensively
            risk_level: mon.capacity_level || 'UNKNOWN',
            current_demand: mon.patient_inflow || 'Normal',
            predicted_demand: mon.predicted_inflow || 'Normal',
            staff_capacity: mon.staff_capacity || 'N/A',
            operating_rooms: mon.operating_rooms_capacity || 'N/A',
            diagnostics: mon.diagnostics_capacity || 'N/A',
            receiving_capacity: mon.ambulance_queue > 0 ? 'CONGESTED' : 'OPEN'
          };
        });

        setNetworkData(merged);
        if (merged.length > 0) setSelectedHospital(merged[0]);
      } catch (err) {
        console.error('[HOSPITAL UI] Failed to load capacity data:', err);
        setError(true);
      } finally {
        setLoading(false);
      }
    };

    fetchCapacityData();
    const interval = setInterval(fetchCapacityData, 30000);
    return () => clearInterval(interval);
  }, []);

  // Derived Filters
  const uniqueAreas = useMemo(() => ['All', ...new Set(networkData.map(h => h.area).filter(Boolean))].sort(), [networkData]);
  const riskLevels = ['All', 'LOW', 'MODERATE', 'HIGH', 'CRITICAL'];
  
  const filteredData = useMemo(() => {
    return networkData.filter(h => {
      if (filterArea !== 'All' && h.area !== filterArea) return false;
      if (filterRisk !== 'All' && h.risk_level !== filterRisk) return false;
      if (searchQuery && !h.name.toLowerCase().includes(searchQuery.toLowerCase())) return false;
      return true;
    }).sort((a, b) => {
      const riskWeight = { CRITICAL: 4, HIGH: 3, MODERATE: 2, LOW: 1, UNKNOWN: 0 };
      return (riskWeight[b.risk_level] || 0) - (riskWeight[a.risk_level] || 0);
    });
  }, [networkData, filterArea, filterRisk, searchQuery]);

  const renderCapacityBar = (label, current, total, invertColor = false) => {
    if (current === 'N/A' || !total) return (
      <div className="flex flex-col gap-1 w-full mb-3">
        <div className="flex justify-between text-[10px] uppercase font-bold text-slate-500 mb-1">
          <span>{label}</span>
          <span>DATA UNAVAILABLE</span>
        </div>
        <div className="flex h-1.5 w-full rounded-full overflow-hidden bg-slate-800"></div>
      </div>
    );
    
    const percent = Math.min((current / total) * 100, 100);
    const colorClass = invertColor 
      ? (percent < 20 ? 'bg-red-500' : percent < 50 ? 'bg-orange-500' : 'bg-emerald-500')
      : (percent > 90 ? 'bg-red-500' : percent > 75 ? 'bg-orange-500' : 'bg-emerald-500');

    return (
      <div className="flex flex-col w-full mb-3">
        <div className="flex justify-between text-[10px] uppercase font-bold text-slate-400 mb-1">
          <span>{label}</span>
          <span>{current} / {total}</span>
        </div>
        <div className="flex h-1.5 w-full rounded-full overflow-hidden bg-slate-800 border border-slate-700">
          <div className={`transition-all duration-500 ${colorClass}`} style={{ width: `${percent}%` }}></div>
        </div>
      </div>
    );
  };

  if (loading) return <div className="flex h-full items-center justify-center font-mono text-slate-400 animate-pulse">Computing Network Capacity Projections...</div>;
  if (error) return <div className="p-6 text-red-400 font-bold uppercase tracking-widest text-sm">Error connecting to Hospital Capacity API</div>;

  return (
    <div className="flex flex-col gap-6 h-full p-6 text-slate-200 bg-slate-950 overflow-y-auto font-sans">
      
      <header className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4 shrink-0 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-3xl font-bold uppercase tracking-tight text-white">Hospital Capacity</h1>
          <p className="text-sm text-slate-400 mt-1">Predictive Demand & Infrastructure Pressure Tracking</p>
        </div>
        <div className="flex items-center gap-3 text-xs font-bold uppercase">
          <span className="px-3 py-1.5 bg-blue-900/30 border border-blue-500 text-blue-400 rounded flex items-center gap-2">
            <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-ping"></span> HOSPITAL AGENT ACTIVE
          </span>
        </div>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 flex-1">
        
        {/* Left: Capacity Network Table */}
        <div className="xl:col-span-8 flex flex-col bg-slate-900 rounded-xl border border-slate-800 overflow-hidden shadow-lg h-full">
          <div className="p-4 border-b border-slate-800 bg-slate-800/30 flex justify-between items-center gap-4 flex-wrap">
            <input 
              type="text" 
              placeholder="Search hospital..." 
              value={searchQuery} 
              onChange={(e) => setSearchQuery(e.target.value)} 
              className="bg-slate-950 border border-slate-700 text-xs rounded px-3 py-2 outline-none text-slate-200 min-w-[200px]"
            />
            <select 
              value={filterArea} 
              onChange={(e) => setFilterArea(e.target.value)} 
              className="bg-slate-950 border border-slate-700 text-xs rounded px-2 py-2 outline-none text-slate-200"
            >
              <option value="" disabled>Filter Area</option>
              {uniqueAreas.map(a => <option key={a} value={a}>{a}</option>)}
            </select>
            <select 
              value={filterRisk} 
              onChange={(e) => setFilterRisk(e.target.value)} 
              className="bg-slate-950 border border-slate-700 text-xs rounded px-2 py-2 outline-none text-slate-200"
            >
              <option value="" disabled>Filter Risk Level</option>
              {riskLevels.map(r => <option key={r} value={r}>{r}</option>)}
            </select>
          </div>
          
          <div className="overflow-y-auto flex-1 custom-scrollbar">
            <table className="w-full text-left text-sm whitespace-nowrap min-w-[800px]">
              <thead className="bg-slate-950/80 text-[10px] uppercase tracking-widest text-slate-500 sticky top-0 border-b border-slate-800">
                <tr>
                  <th className="px-4 py-3">Hospital</th>
                  <th className="px-4 py-3">Current Demand</th>
                  <th className="px-4 py-3">Predicted Demand</th>
                  <th className="px-4 py-3">Projected Pressure</th>
                  <th className="px-4 py-3">Risk Level</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {filteredData.map(h => (
                  <tr 
                    key={h.hospital_id} 
                    onClick={() => setSelectedHospital(h)}
                    className={`cursor-pointer transition-colors ${selectedHospital?.hospital_id === h.hospital_id ? 'bg-blue-900/20' : 'hover:bg-slate-800/50'}`}
                  >
                    <td className={`px-4 py-3 font-bold text-slate-200 border-l-2 ${selectedHospital?.hospital_id === h.hospital_id ? 'border-blue-500 text-blue-400' : 'border-transparent'}`}>{h.name}</td>
                    <td className="px-4 py-3 text-slate-300 font-bold uppercase text-[10px]">{h.current_demand}</td>
                    <td className="px-4 py-3 text-slate-400 font-bold uppercase text-[10px]">{h.predicted_demand}</td>
                    <td className="px-4 py-3 font-mono text-[11px] text-slate-300">{h.capacity_level === 'CRITICAL' ? 'SEVERE CAPACITY LIMIT' : 'WITHIN THRESHOLDS'}</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-1 text-[9px] tracking-wider uppercase font-bold rounded ${h.risk_level === 'CRITICAL' ? 'bg-red-900/50 text-red-400' : h.risk_level === 'HIGH' ? 'bg-orange-900/50 text-orange-400' : 'bg-emerald-900/50 text-emerald-400'}`}>
                        {h.risk_level}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right: Hospital Capacity Detail Panel */}
        <div className="xl:col-span-4 flex flex-col gap-4">
          {selectedHospital ? (
             <div className="bg-slate-900 border border-slate-700 rounded-xl p-5 shadow-lg relative flex flex-col h-full overflow-y-auto custom-scrollbar">
                <div className="absolute top-0 right-0 bg-blue-900/30 text-blue-400 text-[10px] px-3 py-1 font-mono border-bl border-slate-700 rounded-bl-lg uppercase">
                  {selectedHospital.risk_level} RISK
                </div>
                
                <h2 className="text-xl font-bold text-white uppercase pr-16">{selectedHospital.name}</h2>
                <div className="flex gap-3 text-xs text-slate-400 font-mono mt-1 mb-6">
                  <span>{selectedHospital.area}</span>
                  <span>•</span>
                  <span className="uppercase text-slate-300">{selectedHospital.status}</span>
                </div>

                <div className="grid grid-cols-2 gap-3 mb-6">
                  <div className="bg-slate-950 p-3 rounded border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold block mb-1">Current Demand</span>
                    <span className="text-sm font-bold text-slate-200 uppercase">{selectedHospital.current_demand}</span>
                  </div>
                  <div className="bg-slate-950 p-3 rounded border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold block mb-1">Predicted Demand</span>
                    <span className="text-sm font-bold text-blue-400 uppercase">{selectedHospital.predicted_demand}</span>
                  </div>
                </div>

                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 border-b border-slate-800 pb-2">Infrastructure Capacity</h3>
                
                <div className="flex flex-col gap-2 mb-6">
                  {renderCapacityBar('Total Beds Occupied', selectedHospital.occupied_beds, selectedHospital.total_beds)}
                  {renderCapacityBar('ICU Capacity Occupied', selectedHospital.icu_occupied, selectedHospital.icu_total)}
                  {renderCapacityBar('Emergency Bay Utilization', selectedHospital.emergency_occupancy, selectedHospital.emergency_capacity)}
                  {renderCapacityBar('Operating Rooms Active', selectedHospital.operating_rooms, selectedHospital.operating_rooms)}
                  {renderCapacityBar('Diagnostics Availability', selectedHospital.diagnostics, selectedHospital.diagnostics, true)}
                </div>

                <div className="grid grid-cols-2 gap-3 mb-6">
                  <div className="bg-slate-950 p-3 rounded border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold block mb-1">Staff Capacity</span>
                    <span className="text-sm font-mono text-slate-300 uppercase">{selectedHospital.staff_capacity}</span>
                  </div>
                  <div className="bg-slate-950 p-3 rounded border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold block mb-1">Receiving Capacity</span>
                    <span className={`text-sm font-mono font-bold uppercase ${selectedHospital.receiving_capacity === 'CONGESTED' ? 'text-red-400' : 'text-emerald-400'}`}>{selectedHospital.receiving_capacity}</span>
                  </div>
                </div>

                <div className="mt-auto bg-blue-900/10 border border-blue-500/30 p-4 rounded-lg flex flex-col gap-2">
                  <span className="text-[10px] text-blue-400 uppercase tracking-widest font-bold flex items-center gap-2">
                    <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse"></span>
                    Hospital Agent Intelligence
                  </span>
                  <div className="text-sm text-slate-200 font-medium leading-relaxed">
                    {selectedHospital.risk_level === 'CRITICAL' 
                      ? `Projected demand exceeds capacity thresholds. Recommend diverting non-critical inbound flow to alternative regional centers.` 
                      : `Facility possesses adequate capacity to absorb predicted inflow. Maintain standard receiving operations.`}
                  </div>
                </div>
             </div>
          ) : (
            <div className="bg-slate-900 border border-slate-700 rounded-xl p-5 shadow-lg flex items-center justify-center text-slate-500 italic h-full">
              Select a facility to view detailed capacity projections.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}