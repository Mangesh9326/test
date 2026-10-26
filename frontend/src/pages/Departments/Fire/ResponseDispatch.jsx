// frontend/src/pages/Departments/Fire/ResponseDispatch.jsx
import React, { useEffect, useState } from 'react';
import { useFireDispatchStore } from '../../../store/useFireDispatchStore';
import FireOperationsMap from './components/FireOperationsMap';

const SeverityColors = {
  CRITICAL: 'text-red-500 bg-red-500/10 border-red-500/30',
  HIGH: 'text-orange-400 bg-orange-400/10 border-orange-400/30',
  MODERATE: 'text-yellow-400 bg-yellow-400/10 border-yellow-400/30',
  LOW: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/30'
};

const StatusColors = {
  REPORTED: 'text-gray-400',
  ACKNOWLEDGED: 'text-blue-400',
  DISPATCHED: 'text-indigo-400',
  EN_ROUTE: 'text-purple-400',
  ON_SCENE: 'text-orange-400 animate-pulse',
  CONTAINED: 'text-yellow-400',
  RESOLVED: 'text-emerald-400'
};

export default function ResponseDispatch() {
  const {
    queue,
    summary,
    stations,
    filters,
    setFilters,
    selectedIncident,
    setSelectedIncident,
    fetchData,
    isLoading
  } = useFireDispatchStore();

  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    fetchData();
    const dataInterval = setInterval(fetchData, 10000);
    const timeInterval = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => {
      clearInterval(dataInterval);
      clearInterval(timeInterval);
    };
  }, [fetchData]);

  const handleFilterChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFilters({ [name]: type === 'checkbox' ? checked : value });
  };

  const formatTime = (isoString) => {
    if (!isoString) return 'PENDING';
    return new Date(isoString).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
  };

  const calculateDuration = (start, end) => {
    if (!start || !end) return '--';
    const diffMs = new Date(end) - new Date(start);
    return `${Math.round(diffMs / 60000)}m`;
  };

  return (
    <div className="flex flex-col h-full bg-city-900 text-gray-100 p-4 gap-4 overflow-hidden font-sans relative">
      
      {/* HEADER */}
      <div className="flex flex-col gap-3 shrink-0">
        <div className="flex justify-between items-center border-b border-city-700/50 pb-2">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-orange-500">FIRE RESPONSE & DISPATCH</h1>
            <p className="text-xs text-gray-400 font-mono mt-1">Emergency response coordination and unit deployment</p>
          </div>
          <div className="text-[10px] font-mono bg-red-500/10 text-red-400 border border-red-500/30 px-3 py-1.5 rounded-md uppercase tracking-widest flex items-center gap-2 shadow-lg">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
            DEMO / SIMULATED DATA
          </div>
        </div>

        {/* KPIs */}
        <div className="grid grid-cols-6 gap-3">
          {[
            { label: 'ACTIVE INCIDENTS', value: summary.active_incidents || 0, color: 'text-blue-400' },
            { label: 'DISPATCH PENDING', value: summary.dispatch_pending || 0, color: 'text-red-400' },
            { label: 'UNITS DISPATCHED', value: summary.total_tenders_dispatched || 0, color: 'text-indigo-400' },
            { label: 'EN ROUTE', value: summary.en_route || 0, color: 'text-purple-400' },
            { label: 'ON SCENE', value: summary.on_scene || 0, color: 'text-orange-400' },
            { label: 'AVG RESPONSE TIME', value: `${summary.avg_response_time || 0}m`, color: 'text-gray-300' }
          ].map((kpi, idx) => (
            <div key={idx} className="glass-panel p-3 border-city-700/60 flex flex-col justify-between shadow-md bg-city-800/40 border-l-2" style={{ borderLeftColor: 'currentColor', color: kpi.color.replace('text-', '') }}>
              <span className="text-[9px] text-gray-400 font-bold uppercase tracking-widest">{kpi.label}</span>
              <span className={`text-xl font-mono font-bold mt-1 ${kpi.color}`}>{kpi.value}</span>
            </div>
          ))}
        </div>

        {/* FILTERS */}
        <div className="flex gap-3 bg-city-800/30 p-2 border border-city-700/50 rounded-lg items-center">
          <input 
            name="search" 
            placeholder="Search incident..." 
            value={filters.search} 
            onChange={handleFilterChange}
            className="bg-black/50 border border-city-700 rounded px-3 py-1.5 text-xs text-gray-200 outline-none w-48 font-mono"
          />
          <select name="severity" value={filters.severity} onChange={handleFilterChange} className="bg-black/50 border border-city-700 rounded px-2 py-1.5 text-xs text-gray-200 outline-none font-mono">
            <option value="ALL">Severity: All</option>
            <option value="CRITICAL">Critical</option>
            <option value="HIGH">High</option>
            <option value="MODERATE">Moderate</option>
            <option value="LOW">Low</option>
          </select>
          <select name="status" value={filters.status} onChange={handleFilterChange} className="bg-black/50 border border-city-700 rounded px-2 py-1.5 text-xs text-gray-200 outline-none font-mono">
            <option value="ALL">Status: All</option>
            <option value="REPORTED">Dispatch Pending (Reported)</option>
            <option value="DISPATCHED">Dispatched</option>
            <option value="EN_ROUTE">En Route</option>
            <option value="ON_SCENE">On Scene</option>
          </select>
          <label className="flex items-center gap-2 text-xs text-gray-300 font-mono ml-auto">
            <input 
              type="checkbox" 
              name="active_only" 
              checked={filters.active_only} 
              onChange={handleFilterChange} 
              className="rounded bg-black/50 border-city-700 text-orange-500 focus:ring-orange-500"
            />
            ACTIVE ONLY
          </label>
        </div>
      </div>

      {/* MAIN CONTENT GRID */}
      <div className="flex-1 min-h-0 flex flex-col lg:flex-row gap-4 relative">
        
        {/* LEFT COLUMN: Queue & Workload */}
        <div className="flex-[2] flex flex-col gap-4 min-w-0">
          
          {/* QUEUE */}
          <div className="flex-1 glass-panel flex flex-col border-city-700/60 overflow-hidden bg-city-800/20">
            <div className="bg-city-800/80 p-2.5 border-b border-city-700 grid grid-cols-12 text-[10px] font-bold text-gray-400 uppercase tracking-widest gap-2">
              <div className="col-span-2">Incident</div>
              <div className="col-span-2">Location</div>
              <div className="col-span-3">Required Units</div>
              <div className="col-span-2">Station</div>
              <div className="col-span-2">Status</div>
              <div className="col-span-1 text-right">Response</div>
            </div>
            <div className="overflow-y-auto custom-scrollbar flex-1">
              {isLoading && queue.length === 0 ? (
                <div className="p-8 text-center text-xs text-gray-500 font-mono">Loading dispatch queue...</div>
              ) : queue.map(inc => (
                <div 
                  key={inc.id} 
                  onClick={() => setSelectedIncident(inc)}
                  className={`grid grid-cols-12 p-3 border-b border-city-700/30 text-xs items-center cursor-pointer transition-colors gap-2 hover:bg-city-700/40 ${selectedIncident?.id === inc.id ? 'bg-city-700/60 border-l-2 border-l-orange-500' : ''}`}
                >
                  <div className="col-span-2 flex flex-col">
                    <span className="font-mono font-bold text-gray-200">{inc.incident_number}</span>
                    <span className={`text-[9px] w-fit px-1 mt-1 rounded font-bold border ${SeverityColors[inc.severity]}`}>{inc.severity}</span>
                  </div>
                  <div className="col-span-2 flex flex-col">
                    <span className="text-gray-200 truncate">{inc.location}</span>
                    <span className="text-[10px] text-gray-500 truncate">{inc.incident_type.replace('_', ' ')}</span>
                  </div>
                  <div className="col-span-3 flex flex-col text-[10px] font-mono text-gray-400">
                    <div>{inc.fire_tenders_required} Tenders / {inc.rescue_units_dispatched || 0} Rescue</div>
                    <div className={inc.fire_tenders_dispatched < inc.fire_tenders_required ? 'text-red-400 font-bold' : 'text-emerald-400'}>
                      Dispatched: {inc.fire_tenders_dispatched} / {inc.fire_tenders_required}
                    </div>
                  </div>
                  <div className="col-span-2 flex items-center">
                    <span className="text-[10px] text-blue-300 font-mono truncate">{inc.station_name || 'Pending'}</span>
                  </div>
                  <div className="col-span-2">
                    <span className={`font-mono text-[10px] font-bold px-2 py-1 rounded bg-black/40 border border-city-700 ${StatusColors[inc.status]}`}>
                      {inc.status}
                    </span>
                  </div>
                  <div className="col-span-1 text-right font-mono text-gray-400 text-[10px]">
                    {inc.response_time_minutes ? `${inc.response_time_minutes}m` : calculateDuration(inc.reported_at, currentTime)}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* BOTTOM METRICS */}
          <div className="h-48 shrink-0 flex gap-4">
            
            <div className="flex-1 glass-panel border-city-700/60 bg-city-800/20 flex flex-col overflow-hidden">
              <div className="bg-city-800/80 p-2 border-b border-city-700 text-[10px] font-bold text-gray-400 uppercase tracking-widest">
                Fire Station Workload
              </div>
              <div className="overflow-y-auto custom-scrollbar p-2 flex flex-col gap-1">
                {stations.map(st => (
                  <div key={st.station_name} className="flex justify-between items-center p-2 bg-black/30 border border-city-700/30 rounded text-xs">
                    <span className="font-mono text-blue-300 truncate w-1/3">{st.station_name.replace('Fire Station', 'FS')}</span>
                    <span className="text-gray-400 font-mono text-[10px]"><span className="text-orange-400 font-bold">{st.active_incidents}</span> Active</span>
                    <span className="text-gray-400 font-mono text-[10px]"><span className="text-indigo-400 font-bold">{st.dispatched_units || 0}</span> Units</span>
                    <span className="text-gray-400 font-mono text-[10px]"><span className="text-yellow-400 font-bold">{st.on_scene_incidents}</span> On Scene</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="flex-1 glass-panel border-city-700/60 bg-city-800/20 flex flex-col overflow-hidden">
              <div className="bg-city-800/80 p-2 border-b border-city-700 text-[10px] font-bold text-gray-400 uppercase tracking-widest">
                Resource Utilization
              </div>
              <div className="p-4 grid grid-cols-3 gap-4 h-full items-center text-center">
                <div className="flex flex-col gap-1">
                  <span className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">Fire Tenders</span>
                  <div className="text-xl font-mono text-gray-200">
                    <span className={summary.total_tenders_dispatched < summary.total_tenders_required ? 'text-red-400' : 'text-emerald-400'}>
                      {summary.total_tenders_dispatched || 0}
                    </span> 
                    <span className="text-gray-600 text-sm mx-1">/</span> 
                    {summary.total_tenders_required || 0}
                  </div>
                  <span className="text-[9px] text-gray-500">DISPATCHED / REQ</span>
                </div>
                <div className="flex flex-col gap-1 border-x border-city-700/50">
                  <span className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">Rescue Units</span>
                  <div className="text-xl font-mono text-gray-200">{summary.total_rescue_dispatched || 0}</div>
                  <span className="text-[9px] text-gray-500">ACTIVE DEPLOYMENT</span>
                </div>
                <div className="flex flex-col gap-1">
                  <span className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">Water Tankers</span>
                  <div className="text-xl font-mono text-gray-200">{summary.total_tankers_dispatched || 0}</div>
                  <span className="text-[9px] text-gray-500">ACTIVE DEPLOYMENT</span>
                </div>
              </div>
            </div>

          </div>
        </div>

        {/* RIGHT COLUMN: Map */}
        <div className="flex-1 flex flex-col glass-panel border-city-700/60 p-1 relative overflow-hidden">
          <div className="absolute top-2 left-2 z-10 bg-black/80 px-3 py-2 rounded border border-city-700 backdrop-blur-sm pointer-events-none">
            <span className="text-[10px] font-bold text-gray-300 uppercase tracking-widest block mb-2">Response Map Legend</span>
            <div className="flex flex-col gap-1.5 text-[9px] font-mono">
              <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-yellow-500"></span> DISPATCH PENDING</div>
              <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-blue-500"></span> DISPATCHED</div>
              <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-purple-500"></span> EN ROUTE</div>
              <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-red-500"></span> ON SCENE</div>
              <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-emerald-500"></span> CONTAINED</div>
            </div>
          </div>
          <FireOperationsMap incidents={queue} selectedIncident={selectedIncident} onSelectIncident={setSelectedIncident} />
        </div>
      </div>

      {/* DISPATCH DRAWER */}
      <div className={`absolute top-0 right-0 h-full w-[450px] glass-panel bg-city-900 border-l border-city-700 shadow-2xl transition-transform duration-300 z-50 flex flex-col ${selectedIncident ? 'translate-x-0' : 'translate-x-full'}`}>
        {selectedIncident && (
          <>
            <div className="p-4 border-b border-city-700/50 flex justify-between items-start bg-black/40">
              <div>
                <div className="text-[10px] font-mono text-gray-400 mb-1">INCIDENT {selectedIncident.incident_number}</div>
                <h2 className="text-lg font-bold text-gray-100">{selectedIncident.incident_type.replace('_', ' ')}</h2>
                <div className="flex gap-2 mt-2">
                  <span className={`text-[9px] px-2 py-0.5 rounded font-bold border ${SeverityColors[selectedIncident.severity]}`}>{selectedIncident.severity} SEVERITY</span>
                  <span className={`text-[9px] px-2 py-0.5 rounded font-bold border border-gray-600 bg-gray-800 text-gray-300`}>{selectedIncident.priority} PRIORITY</span>
                </div>
              </div>
              <button onClick={() => setSelectedIncident(null)} className="text-gray-500 hover:text-white bg-black/50 rounded p-1.5 border border-city-700">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
              </button>
            </div>

            <div className="p-5 flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-6">
              
              <div className="grid grid-cols-2 gap-y-3 gap-x-4 text-xs">
                <div><span className="text-gray-500 block text-[10px] mb-0.5">LOCATION</span><span className="text-gray-200">{selectedIncident.location}</span></div>
                <div><span className="text-gray-500 block text-[10px] mb-0.5">WARD</span><span className="text-gray-200">{selectedIncident.ward}</span></div>
                <div className="col-span-2"><span className="text-gray-500 block text-[10px] mb-0.5">ADDRESS</span><span className="text-gray-200">{selectedIncident.address}</span></div>
                <div className="col-span-2"><span className="text-gray-500 block text-[10px] mb-0.5">LANDMARK</span><span className="text-gray-200">{selectedIncident.landmark}</span></div>
              </div>

              {/* RESOURCE ALLOCATION */}
              <div className="bg-black/30 border border-city-700 rounded-lg p-3">
                <h3 className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-3 flex justify-between">
                  <span>Resource Allocation</span>
                  <span className="text-blue-300">{selectedIncident.station_name || 'Station Pending'}</span>
                </h3>
                
                <div className="flex flex-col gap-3 font-mono text-xs">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Fire Tenders</span>
                    <div className="flex items-center gap-4">
                      <span>Req: {selectedIncident.fire_tenders_required}</span>
                      <span className={selectedIncident.fire_tenders_dispatched < selectedIncident.fire_tenders_required ? 'text-red-400 font-bold' : 'text-emerald-400'}>Disp: {selectedIncident.fire_tenders_dispatched}</span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Rescue Units</span>
                    <span className="text-gray-200">Disp: {selectedIncident.rescue_units_dispatched}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Water Tankers</span>
                    <span className="text-gray-200">Disp: {selectedIncident.water_tankers_dispatched}</span>
                  </div>
                </div>

                {selectedIncident.fire_tenders_dispatched < selectedIncident.fire_tenders_required && (
                  <div className="mt-4 bg-red-500/10 border border-red-500/30 p-2 rounded text-red-400 text-[10px] flex items-start gap-2">
                    <svg className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
                    <div>
                      <span className="font-bold uppercase block mb-0.5">Fire Tender Shortage</span>
                      Required: {selectedIncident.fire_tenders_required} | Dispatched: {selectedIncident.fire_tenders_dispatched} | Shortage: {selectedIncident.fire_tenders_required - selectedIncident.fire_tenders_dispatched}
                    </div>
                  </div>
                )}
              </div>

              {/* TIMELINE */}
              <div>
                <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest border-b border-city-700 pb-1 mb-3">Response Timeline</h3>
                <div className="flex flex-col gap-2 font-mono text-xs relative before:absolute before:inset-y-2 before:left-[4.5px] before:w-[1px] before:bg-city-700">
                  {[
                    { label: 'REPORTED', time: selectedIncident.reported_at, active: !!selectedIncident.reported_at },
                    { label: 'ACKNOWLEDGED', time: selectedIncident.acknowledged_at, active: !!selectedIncident.acknowledged_at },
                    { label: 'DISPATCHED', time: selectedIncident.dispatched_at, active: !!selectedIncident.dispatched_at },
                    { label: 'EN ROUTE', time: selectedIncident.en_route_at || (['EN_ROUTE','ON_SCENE','CONTAINED','RESOLVED'].includes(selectedIncident.status) ? selectedIncident.dispatched_at : null), active: ['EN_ROUTE','ON_SCENE','CONTAINED','RESOLVED'].includes(selectedIncident.status) },
                    { label: 'ON SCENE', time: selectedIncident.arrived_at, active: !!selectedIncident.arrived_at },
                    { label: 'CONTAINED', time: selectedIncident.contained_at, active: !!selectedIncident.contained_at },
                    { label: 'RESOLVED', time: selectedIncident.resolved_at, active: !!selectedIncident.resolved_at },
                  ].map((step, i) => (
                    <div key={i} className={`flex items-center justify-between pl-4 relative ${step.active ? 'text-gray-200' : 'text-gray-600'}`}>
                      <div className={`absolute left-[1px] top-1.5 w-2 h-2 rounded-full border border-city-900 ${step.active ? 'bg-blue-400' : 'bg-city-700'}`}></div>
                      <span>{step.label}</span>
                      <span>{formatTime(step.time)}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* ACTION BUTTONS (SIMULATED) */}
              <div className="mt-auto pt-4 flex flex-col gap-2">
                <p className="text-[9px] text-gray-500 text-center font-mono uppercase tracking-widest">Simulated Dispatch Controls</p>
                <div className="grid grid-cols-2 gap-2">
                  <button className="bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold py-2 rounded transition-colors">ASSIGN UNIT</button>
                  <button className="bg-red-600/20 text-red-400 hover:bg-red-600/40 border border-red-500/30 text-xs font-bold py-2 rounded transition-colors">ESCALATE</button>
                  <button className="bg-city-800 hover:bg-city-700 text-gray-300 text-xs font-bold py-2 rounded transition-colors col-span-2">CENTER MAP ON INCIDENT</button>
                </div>
              </div>

            </div>
          </>
        )}
      </div>
    </div>
  );
}