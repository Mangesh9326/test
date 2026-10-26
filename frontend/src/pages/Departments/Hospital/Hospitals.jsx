import React, { useState, useEffect, useMemo, useCallback } from 'react';
import HospitalTable from './components/HospitalTable';
import HospitalProfileModal from './components/HospitalProfileModal';
import HospitalDetailDrawer from './components/HospitalDetailDrawer';

export default function Hospitals() {
  const [hospitals, setHospitals] = useState([]);
  const [selectedHospital, setSelectedHospital] = useState(null);
  const [isProfileModalOpen, setIsProfileModalOpen] = useState(false);
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  // Filters State
  const [searchQuery, setSearchQuery] = useState('');
  const [filterArea, setFilterArea] = useState('All');
  const [filterType, setFilterType] = useState('All');
  const [filterEmergency, setFilterEmergency] = useState('All');
  const [filterICU, setFilterICU] = useState('All');
  const [filterPressure, setFilterPressure] = useState('All');
  const [filterStatus, setFilterStatus] = useState('All');
  const [filterInflow, setFilterInflow] = useState('All');

  // Prevent memory leaks and duplicate overlapping requests
  useEffect(() => {
    let isMounted = true;
    let timerId;
    const controller = new AbortController();

    const fetchHospitalData = async () => {
      try {
        const [hospitalsRes, monitoringRes] = await Promise.all([
          fetch('/api/hospitals?limit=100', { signal: controller.signal }),
          fetch('/api/hospitals/monitoring', { signal: controller.signal })
        ]);

        if (!hospitalsRes.ok || !monitoringRes.ok) throw new Error('API Error');

        const hospitalsData = await hospitalsRes.json();
        const monitoringData = await monitoringRes.json();

        const merged = hospitalsData.data.map(base => {
          const mon = monitoringData.find(m => m.hospital_id === base.hospital_id) || {};
          return { ...base, ...mon };
        });

        if (isMounted) {
          setHospitals(merged);
          setLastUpdated(new Date());
          setError(null);
        }
      } catch (err) {
        if (isMounted && err.name !== 'AbortError') {
          console.error("[HOSPITAL UI] Error fetching networks:", err);
          setError("Unable to synchronize hospital network data.");
        }
      } finally {
        if (isMounted) {
          setLoading(false);
          // Chain timeout only after previous fetch finishes
          timerId = setTimeout(fetchHospitalData, 30000);
        }
      }
    };

    fetchHospitalData();

    return () => {
      isMounted = false;
      clearTimeout(timerId);
      controller.abort();
    };
  }, []);

  const handleSelectHospital = useCallback(async (id) => {
    if (!id) {
       setSelectedHospital(null);
       return;
    }
    try {
      // Fetch fresh, full-profile data + specific agent details
      const [detailRes, agentRes] = await Promise.all([
        fetch(`/api/hospitals/${id}`),
        fetch(`/api/hospital/agent/hospital/${id}`)
      ]);
      
      if (!detailRes.ok) throw new Error('Failed to fetch hospital details');
      
      const data = await detailRes.json();
      let agentData = null;
      if (agentRes.ok) {
         agentData = await agentRes.json();
      }
      
      setSelectedHospital({ ...data, agent: agentData });
    } catch (err) {
      console.error("[HOSPITAL UI] Detail/Agent fetch failed:", err);
    }
  }, []);

  const clearFilters = () => {
    setSearchQuery(''); setFilterArea('All'); setFilterType('All'); setFilterEmergency('All');
    setFilterICU('All'); setFilterPressure('All'); setFilterStatus('All'); setFilterInflow('All');
  };

  const uniqueAreas = useMemo(() => ['All', ...new Set(hospitals.map(h => h.area).filter(Boolean))].sort(), [hospitals]);
  const typeOptions = ['All', 'Public', 'Private', 'Trust', 'Specialty'];
  const pressureOptions = ['All', 'Normal', 'Moderate', 'High', 'Critical'];
  const statusOptions = ['All', 'Active', 'Standby', 'Restricted', 'Offline'];
  const inflowOptions = ['All', 'Normal', 'Increasing', 'High', 'Critical'];
  const emergencyOptions = ['All', 'Available', 'Unavailable'];
  const icuOptions = ['All', 'Normal', 'High Pressure', 'Critical'];

  const filteredHospitals = useMemo(() => {
    return hospitals.filter(h => {
      if (searchQuery) {
        const q = searchQuery.toLowerCase();
        const matchesSearch = (h.name || '').toLowerCase().includes(q) || (h.area || '').toLowerCase().includes(q) || (h.hospital_type || '').toLowerCase().includes(q);
        if (!matchesSearch) return false;
      }
      if (filterArea !== 'All' && h.area !== filterArea) return false;
      if (filterType !== 'All' && !(h.hospital_type || '').toLowerCase().includes(filterType.toLowerCase())) return false;
      if (filterStatus !== 'All' && (h.status || 'Active').toLowerCase() !== filterStatus.toLowerCase()) return false;
      if (filterEmergency !== 'All') {
        const hasEmer = filterEmergency === 'Available';
        if (h.emergency_available !== hasEmer) return false;
      }
      if (filterPressure !== 'All' && (h.capacity_level || 'NORMAL') !== filterPressure.toUpperCase()) return false;
      if (filterInflow !== 'All' && (h.patient_inflow || 'NORMAL') !== filterInflow.toUpperCase()) return false;
      if (filterICU !== 'All') {
        const icuRatio = (h.icu_occupied || 0) / (h.icu_total || 1);
        const icuAvail = h.icu_available || 0;
        let icuStatus = 'Normal';
        if (icuAvail <= 5 || icuRatio >= 0.90) icuStatus = 'Critical';
        else if (icuRatio >= 0.80) icuStatus = 'High Pressure';
        if (filterICU !== icuStatus) return false;
      }
      return true;
    });
  }, [hospitals, searchQuery, filterArea, filterType, filterStatus, filterEmergency, filterPressure, filterInflow, filterICU]);

  const activeFilterCount = useMemo(() => {
    let count = 0;
    if (searchQuery) count++;
    if (filterArea !== 'All') count++;
    if (filterType !== 'All') count++;
    if (filterEmergency !== 'All') count++;
    if (filterICU !== 'All') count++;
    if (filterPressure !== 'All') count++;
    if (filterStatus !== 'All') count++;
    if (filterInflow !== 'All') count++;
    return count;
  }, [searchQuery, filterArea, filterType, filterEmergency, filterICU, filterPressure, filterStatus, filterInflow]);

  const kpis = useMemo(() => {
    let operational = 0, emergencyReady = 0, highPressure = 0, icuAvailable = 0, bedsAvailable = 0;
    hospitals.forEach(h => {
      if (h.status === 'Active') operational++;
      if (h.emergency_available) emergencyReady++;
      if (h.capacity_level === 'HIGH' || h.capacity_level === 'CRITICAL') highPressure++;
      icuAvailable += (h.icu_available || 0);
      bedsAvailable += (h.available_beds || 0);
    });
    const total = hospitals.length;
    let networkCondition = 'NORMAL';
    let conditionColor = 'text-emerald-400';
    if (total > 0) {
      const pressureRatio = highPressure / total;
      if (pressureRatio > 0.3) { networkCondition = 'CRITICAL'; conditionColor = 'text-red-500'; }
      else if (pressureRatio > 0.15) { networkCondition = 'HIGH'; conditionColor = 'text-orange-500'; }
      else if (highPressure > 0) { networkCondition = 'MODERATE'; conditionColor = 'text-yellow-400'; }
    }
    return { total, operational, emergencyReady, highPressure, icuAvailable, bedsAvailable, networkCondition, conditionColor };
  }, [hospitals]);

  const currentIndex = selectedHospital ? filteredHospitals.findIndex(h => h.hospital_id === selectedHospital.hospital_id) : -1;
  const hasPrev = currentIndex > 0;
  const hasNext = currentIndex >= 0 && currentIndex < filteredHospitals.length - 1;

  if (loading && hospitals.length === 0) {
    return (
      <div className="flex h-full items-center justify-center bg-slate-950">
        <div className="flex items-center gap-3">
          <span className="w-2 h-2 bg-blue-500 rounded-full animate-ping"></span>
          <span className="text-slate-400 font-mono text-sm uppercase tracking-widest">Initializing Network Operations Center...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5 h-full text-slate-200 overflow-hidden font-sans relative">
      
      {isProfileModalOpen && selectedHospital && (
        <HospitalProfileModal 
          hospital={selectedHospital} 
          onClose={() => setIsProfileModalOpen(false)} 
          onPrev={() => handleSelectHospital(filteredHospitals[currentIndex - 1].hospital_id)}
          onNext={() => handleSelectHospital(filteredHospitals[currentIndex + 1].hospital_id)}
          hasPrev={hasPrev}
          hasNext={hasNext}
        />
      )}

      {/* Header */}
      <header className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4 shrink-0 border-b border-slate-800/80 pb-4">
        <div>
          <h1 className="text-2xl font-bold uppercase tracking-tight text-white">Mumbai Hospital Network Operations Center</h1>
          <div className="flex flex-wrap items-center gap-3 mt-1.5 text-[10px] font-bold uppercase tracking-widest text-slate-400">
            <span className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.8)]"></span>
              Auto-Refresh Active
            </span>
            <span className="text-slate-600">|</span>
            <span>Last Updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : '...'}</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {error && <span className="px-3 py-1 bg-red-900/20 border border-red-900/50 text-red-400 text-[10px] font-bold uppercase tracking-widest rounded shadow-sm">{error}</span>}
          <span className="px-3 py-1.5 bg-blue-900/20 border border-blue-800/50 text-blue-400 text-[10px] font-bold uppercase tracking-widest rounded flex items-center gap-2 shadow-sm">
            <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse"></span> HOSPITAL AGENT ACTIVE
          </span>
        </div>
      </header>

      {/* KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-7 gap-3 shrink-0">
        <div className="bg-slate-900/80 border border-slate-800 p-3.5 rounded-lg flex flex-col justify-center shadow-sm">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Network Condition</span>
          <span className={`text-lg font-bold uppercase tracking-tight mt-1 ${kpis.conditionColor}`}>{kpis.networkCondition}</span>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 p-3.5 rounded-lg flex flex-col justify-center shadow-sm">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Monitored</span>
          <span className="text-xl font-mono text-slate-200 mt-1">{kpis.total}</span>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 p-3.5 rounded-lg flex flex-col justify-center shadow-sm">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Operational</span>
          <span className="text-xl font-mono text-emerald-400 mt-1">{kpis.operational}</span>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 p-3.5 rounded-lg flex flex-col justify-center shadow-sm">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Emergency Ready</span>
          <span className="text-xl font-mono text-blue-400 mt-1">{kpis.emergencyReady}</span>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 p-3.5 rounded-lg flex flex-col justify-center shadow-sm">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">High Pressure</span>
          <span className={`text-xl font-mono mt-1 ${kpis.highPressure > 0 ? 'text-orange-500' : 'text-slate-200'}`}>{kpis.highPressure}</span>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 p-3.5 rounded-lg flex flex-col justify-center shadow-sm">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">ICU Available</span>
          <span className="text-xl font-mono text-blue-300 mt-1">{kpis.icuAvailable}</span>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 p-3.5 rounded-lg flex flex-col justify-center shadow-sm">
          <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Beds Available</span>
          <span className="text-xl font-mono text-emerald-300 mt-1">{kpis.bedsAvailable}</span>
        </div>
      </div>

      {/* Main Layout Area (Filters + Table + Drawer) */}
      <div className="flex flex-col xl:flex-row flex-1 min-h-0 overflow-hidden relative w-full">
        
        {/* Main Table Column */}
        {/* Uses flex-1 and min-w-0 to allow width transition without breaking Flexbox child layouts */}
        <div className={`flex-1 flex flex-col gap-4 min-w-0 h-full transition-all duration-300 ${selectedHospital ? 'xl:pr-0 hidden xl:flex' : 'flex'}`}>
          <div className="bg-slate-900/50 rounded-xl border border-slate-800 overflow-hidden shadow-sm flex flex-col h-full w-full">
            
            {/* Advanced Filter Bar */}
            <div className="p-3 border-b border-slate-800 bg-slate-900/80 flex flex-col xl:flex-row gap-3 items-start xl:items-center justify-between shrink-0">
              <div className="flex flex-wrap items-center gap-2">
                <input type="text" placeholder="Search hospital, area, type..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} className="bg-slate-950 border border-slate-800 text-xs rounded px-3 py-1.5 outline-none text-slate-200 min-w-[180px] focus:border-slate-600 transition-colors"/>
                <select value={filterArea} onChange={(e) => setFilterArea(e.target.value)} className="bg-slate-950 border border-slate-800 text-xs rounded px-2 py-1.5 outline-none text-slate-300 focus:border-slate-600 transition-colors"><option value="" disabled>Area</option>{uniqueAreas.map(a => <option key={a} value={a}>{a}</option>)}</select>
                <select value={filterType} onChange={(e) => setFilterType(e.target.value)} className="bg-slate-950 border border-slate-800 text-xs rounded px-2 py-1.5 outline-none text-slate-300 focus:border-slate-600 transition-colors"><option value="" disabled>Hospital Type</option>{typeOptions.map(t => <option key={t} value={t}>{t}</option>)}</select>
                <select value={filterPressure} onChange={(e) => setFilterPressure(e.target.value)} className="bg-slate-950 border border-slate-800 text-xs rounded px-2 py-1.5 outline-none text-slate-300 focus:border-slate-600 transition-colors"><option value="" disabled>Pressure</option>{pressureOptions.map(p => <option key={p} value={p}>{p}</option>)}</select>
                <select value={filterStatus} onChange={(e) => setFilterStatus(e.target.value)} className="bg-slate-950 border border-slate-800 text-xs rounded px-2 py-1.5 outline-none text-slate-300 focus:border-slate-600 transition-colors"><option value="" disabled>Op Status</option>{statusOptions.map(s => <option key={s} value={s}>{s}</option>)}</select>
                <select value={filterInflow} onChange={(e) => setFilterInflow(e.target.value)} className="bg-slate-950 border border-slate-800 text-xs rounded px-2 py-1.5 outline-none text-slate-300 focus:border-slate-600 transition-colors"><option value="" disabled>Inflow</option>{inflowOptions.map(i => <option key={i} value={i}>{i}</option>)}</select>
                <select value={filterEmergency} onChange={(e) => setFilterEmergency(e.target.value)} className="bg-slate-950 border border-slate-800 text-xs rounded px-2 py-1.5 outline-none text-slate-300 focus:border-slate-600 transition-colors"><option value="" disabled>Emergency</option>{emergencyOptions.map(e => <option key={e} value={e}>{e}</option>)}</select>
                <select value={filterICU} onChange={(e) => setFilterICU(e.target.value)} className="bg-slate-950 border border-slate-800 text-xs rounded px-2 py-1.5 outline-none text-slate-300 focus:border-slate-600 transition-colors"><option value="" disabled>ICU Pressure</option>{icuOptions.map(i => <option key={i} value={i}>{i}</option>)}</select>
              </div>
              
              <div className="flex items-center gap-3 shrink-0">
                {activeFilterCount > 0 && (
                  <span className="text-[10px] text-blue-400 font-bold uppercase tracking-widest bg-blue-900/20 px-2 py-1 rounded border border-blue-800/50">
                    {activeFilterCount} Active Filters
                  </span>
                )}
                <div className="text-[10px] text-slate-400 font-bold uppercase tracking-widest bg-slate-950 px-2 py-1 rounded border border-slate-800">
                  Showing {filteredHospitals.length} of {hospitals.length}
                </div>
                <button onClick={clearFilters} className="text-[9px] font-bold text-slate-300 bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded uppercase tracking-widest whitespace-nowrap transition-colors shadow-sm focus:outline-none focus:ring-2 focus:ring-slate-500">
                  RESET ALL
                </button>
              </div>
            </div>

            <HospitalTable data={filteredHospitals} selectedId={selectedHospital?.hospital_id} onSelect={handleSelectHospital} />
          </div>
        </div>

        {/* Right Panel / Drawer (Hardware Accelerated Animation) */}
        {/* Animated container wrapper dynamically expands/collapses width */}
        <div 
          className={`absolute inset-0 z-20 xl:static xl:z-auto transition-all duration-300 ease-in-out flex shrink-0 h-full overflow-hidden ${
            selectedHospital ? 'xl:w-[400px] xl:ml-5 opacity-100 translate-x-0' : 'xl:w-0 opacity-0 translate-x-full xl:translate-x-0 pointer-events-none'
          }`}
        >
          {/* Inner container stays fixed width to prevent text squishing during animation */}
          <div className="w-full xl:w-[400px] shrink-0 h-full flex flex-col">
             <HospitalDetailDrawer 
               hospital={selectedHospital} 
               onClose={() => handleSelectHospital(null)} 
               onOpenProfile={() => setIsProfileModalOpen(true)} 
             />
          </div>
        </div>

      </div>
    </div>
  );
}