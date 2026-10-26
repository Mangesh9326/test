import React, { useEffect, useState, useMemo } from 'react';
import { useCitizenStore } from '../../store/useCitizenStore';
import DigitalTwinMap from '../../components/DigitalTwin/MapWidget';

export default function AroundMe() {
  const { 
    location, radius, services, incidents, traffic, alerts, 
    isLoading, error, isSimulated, fetchAroundMe, setRadius, setLocation, askAI,
    subscribeToSockets, unsubscribeFromSockets 
  } = useCitizenStore();

  const [activeFilter, setActiveFilter] = useState('ALL');
  const [selectedEntity, setSelectedEntity] = useState(null);
  
  // AI State
  const [aiQuery, setAiQuery] = useState('');
  const [aiResponse, setAiResponse] = useState('');
  const [isAiLoading, setIsAiLoading] = useState(false);
  const [showAiDrawer, setShowAiDrawer] = useState(false);

  useEffect(() => {
    fetchAroundMe();
    subscribeToSockets();
    
    // Attempt real browser geolocation gracefully
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        (pos) => setLocation(pos.coords.latitude, pos.coords.longitude, "Current Location"),
        (err) => console.log("Geolocation unavailable/denied, falling back to Demo location.")
      );
    }
    return () => unsubscribeFromSockets();
  }, []);

  // Stop the page behind a drawer / bottom sheet from scrolling (matters on phones)
  const anyDrawerOpen = Boolean(selectedEntity) || showAiDrawer;
  useEffect(() => {
    if (!anyDrawerOpen) return undefined;
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = prev; };
  }, [anyDrawerOpen]);

  const handleAskAI = async () => {
    if (!aiQuery.trim()) return;
    setIsAiLoading(true);
    const reply = await askAI(aiQuery);
    setAiResponse(reply);
    setIsAiLoading(false);
  };

  // Filter Logic
  const displayItems = useMemo(() => {
    const allItems = [...services, ...incidents];
    if (activeFilter === 'ALL') return allItems;
    if (activeFilter === 'INCIDENTS') return incidents;
    if (activeFilter === 'ALERTS') return alerts;
    return services.filter(s => s.category.toUpperCase() === activeFilter);
  }, [services, incidents, alerts, activeFilter]);

  const severityColor = (severity) => {
    switch (severity?.toUpperCase()) {
      case 'CRITICAL': return 'bg-red-500/20 text-red-500 border-red-500/50';
      case 'HIGH': return 'bg-orange-500/20 text-orange-500 border-orange-500/50';
      case 'MODERATE': return 'bg-yellow-500/20 text-yellow-500 border-yellow-500/50';
      default: return 'bg-emerald-500/20 text-emerald-500 border-emerald-500/50';
    }
  };

  if (error) return <div className="p-8 text-red-400 text-center">{error}</div>;

  return (
    // Phones/tablets: normal page scroll. Desktop (lg+): fixed-height split layout.
    <div className="min-h-full lg:h-full flex flex-col gap-4 lg:gap-6 text-slate-100 font-sans p-3 sm:p-4 lg:p-6 bg-slate-950">
      
      {/* HEADER & CONTROLS */}
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-3 lg:gap-4">
        <div className="min-w-0">
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">Around Me</h1>
          <p className="text-sm text-slate-400 mt-1">
            Understand what's happening within {radius} km of <span className="text-slate-200 font-semibold">{location.name}</span>
          </p>
        </div>

        <div className="grid grid-cols-2 sm:flex sm:flex-wrap gap-2 sm:gap-3 items-center w-full lg:w-auto">
          <select 
            value={radius}
            onChange={(e) => setRadius(Number(e.target.value))}
            aria-label="Search radius"
            className="w-full sm:w-auto bg-slate-900 border border-slate-700 text-sm rounded-lg px-3 py-2.5 text-slate-200 outline-none focus:border-blue-500"
          >
            <option value={1}>1 km Radius</option>
            <option value={3}>3 km Radius</option>
            <option value={5}>5 km Radius</option>
            <option value={10}>10 km Radius</option>
          </select>
          <button 
            onClick={() => setShowAiDrawer(true)}
            className="w-full sm:w-auto bg-purple-600 hover:bg-purple-500 text-white text-sm font-semibold py-2.5 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
            Ask AI
          </button>
        </div>
      </div>

      {/* KPI GRID: 2 cols phone, 3 cols tablet, 5 cols desktop */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 lg:gap-4">
        <KPICard title="Emergency Services" count={services.filter(s => ['Hospital', 'Police', 'Fire'].includes(s.category)).length} color="text-blue-400" />
        <KPICard title="Active Incidents" count={incidents.length} color="text-red-400" />
        <KPICard title="Critical Alerts" count={alerts.length} color="text-orange-400" />
        <KPICard title="Traffic Nodes" count={traffic.length} color="text-yellow-400" />
        <KPICard title="Nearby Utilities" count={services.filter(s => s.category === 'Utility').length} color="text-emerald-400" className="col-span-2 sm:col-span-1" />
      </div>

      {/* MAIN CONTENT SPLIT
          Mobile: map first (full-bleed square), then the list.
          Desktop: list left (1/3), map right (2/3). */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 lg:gap-6 lg:flex-1 lg:min-h-0">
        
        {/* MAP: full-width + square on phones, 16:10 on tablets, fills the column on desktop */}
        <div className="order-1 lg:order-2 lg:col-span-2 relative overflow-hidden bg-slate-900 border-y sm:border border-slate-800 -mx-3 sm:mx-0 rounded-none sm:rounded-xl aspect-square sm:aspect-[16/10] lg:aspect-auto lg:min-h-[400px]">
          <div className="absolute inset-0">
            <DigitalTwinMap 
              focusedCenter={[location.latitude, location.longitude]} 
              focusedZoom={13}
              onEntityClick={(e) => setSelectedEntity(e)}
              // Map markers are driven by the project's existing useMapStore / components.
              // By filtering in backend, we assume the unified map naturally reads the updated context.
            />
          </div>
        </div>

        {/* LIST + FILTERS */}
        <div className="order-2 lg:order-1 lg:col-span-1 flex flex-col gap-3 lg:gap-4 min-h-0 max-h-[70vh] lg:max-h-none bg-slate-900/50 border border-slate-800 rounded-xl p-3 sm:p-4">
          
          {/* Filters scroll sideways on narrow screens, wrap on desktop */}
          <div className="flex gap-2 shrink-0 overflow-x-auto lg:flex-wrap lg:overflow-visible -mx-1 px-1 pb-1 lg:pb-0 custom-scrollbar">
             {['ALL', 'INCIDENTS', 'ALERTS', 'HOSPITAL', 'POLICE', 'FIRE'].map(filter => (
               <button
                 key={filter}
                 onClick={() => setActiveFilter(filter)}
                 className={`shrink-0 whitespace-nowrap px-3 py-2 lg:py-1.5 text-xs font-semibold rounded-full border transition-all ${
                   activeFilter === filter 
                     ? 'bg-blue-600/20 text-blue-400 border-blue-500/50' 
                     : 'bg-slate-800/50 text-slate-400 border-slate-700 hover:bg-slate-800'
                 }`}
               >
                 {filter}
               </button>
             ))}
          </div>

          <div className="flex-1 min-h-[12rem] overflow-y-auto pr-1 sm:pr-2 flex flex-col gap-3 custom-scrollbar">
            {isLoading ? (
              <div className="text-center text-slate-500 mt-10">Loading nearby entities...</div>
            ) : displayItems.length === 0 ? (
              <div className="text-center text-slate-500 mt-10">No records found within {radius} km.</div>
            ) : (
              displayItems.map((item, idx) => (
                <div 
                  key={item.id || idx}
                  onClick={() => setSelectedEntity(item)}
                  className={`p-3 sm:p-4 rounded-lg border cursor-pointer transition-all ${
                    selectedEntity?.id === item.id 
                      ? 'bg-slate-800 border-slate-500' 
                      : 'bg-slate-900/80 border-slate-800 hover:border-slate-600'
                  }`}
                >
                  <div className="flex justify-between items-start gap-2 mb-2">
                    <span className="text-xs font-bold uppercase text-slate-400 tracking-wider">{item.category}</span>
                    <span className="text-xs font-mono text-slate-500 shrink-0">{Number(item.distance).toFixed(1)} km</span>
                  </div>
                  <h3 className="font-semibold text-slate-200 break-words">{item.name}</h3>
                  <div className="flex justify-between items-end gap-2 mt-3">
                    <span className="text-xs text-slate-400 truncate min-w-0 flex-1">{item.location}</span>
                    {item.category === 'Incident' ? (
                       <span className={`shrink-0 text-[10px] px-2 py-1 rounded border font-bold uppercase ${severityColor(item.severity)}`}>
                         {item.severity}
                       </span>
                    ) : (
                       <span className={`shrink-0 text-[10px] px-2 py-1 rounded border font-bold uppercase ${
                         item.status?.toUpperCase() === 'OPERATIONAL' || item.status?.toUpperCase() === 'ACTIVE' 
                         ? 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20' 
                         : 'bg-slate-700/50 text-slate-400 border-slate-600'
                       }`}>
                         {item.status || 'UNKNOWN'}
                       </span>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

      </div>

      {/* ENTITY DETAIL: bottom sheet on phones, right-side drawer from sm up */}
      {selectedEntity && (
        <>
          <div className="sm:hidden fixed inset-0 bg-black/60 z-40" onClick={() => setSelectedEntity(null)} aria-hidden="true" />
          <div
            role="dialog" aria-label="Entity details"
            className="fixed z-50 bg-slate-900 border-slate-800 shadow-2xl overflow-y-auto
                       inset-x-0 bottom-0 max-h-[80dvh] rounded-t-2xl border-t p-5 pb-[max(1.25rem,env(safe-area-inset-bottom))]
                       sm:inset-x-auto sm:inset-y-0 sm:right-0 sm:w-80 sm:max-h-none sm:rounded-none sm:border-t-0 sm:border-l sm:pb-5"
          >
            <div className="sm:hidden mx-auto mb-3 h-1 w-10 rounded-full bg-slate-700" aria-hidden="true" />
            <button
              onClick={() => setSelectedEntity(null)}
              aria-label="Close details"
              className="absolute top-3 right-3 sm:top-4 sm:right-4 h-9 w-9 flex items-center justify-center rounded-lg text-slate-400 hover:text-white hover:bg-slate-800"
            >✕</button>
            <h2 className="text-xl font-bold mt-2 sm:mt-6 pr-10 text-slate-100 break-words">{selectedEntity.name}</h2>
            <p className="text-sm text-slate-400 mt-1">{selectedEntity.type}</p>
            
            <div className="mt-5 sm:mt-6 flex flex-col gap-4">
               <DetailRow label="Category" value={selectedEntity.category} />
               <DetailRow label="Distance" value={`${Number(selectedEntity.distance).toFixed(2)} km`} />
               <DetailRow label="Location" value={selectedEntity.location} />
               <DetailRow label="Status" value={selectedEntity.status || 'N/A'} />
               
               {selectedEntity.category === 'Incident' && (
                 <>
                   <DetailRow label="Severity" value={selectedEntity.severity} />
                   <DetailRow label="Reported" value={new Date(selectedEntity.time).toLocaleString()} />
                 </>
               )}
               {selectedEntity.category === 'Hospital' && (
                 <DetailRow label="Emergency Services" value={selectedEntity.emergency_available ? 'Available' : 'Unavailable'} />
               )}
               
               <div className="mt-2 flex gap-2">
                  <button className="flex-1 bg-blue-600 hover:bg-blue-500 text-white text-sm sm:text-xs font-semibold py-3 sm:py-2 rounded transition-colors">
                    View on Map
                  </button>
               </div>
            </div>
          </div>
        </>
      )}

      {/* AI ASSISTANT: full screen on phones, right-side drawer from sm up */}
      {showAiDrawer && (
        <div
          role="dialog" aria-label="Citizen AI"
          className="fixed z-50 flex flex-col bg-slate-950 border-slate-800 shadow-[0_0_50px_rgba(147,51,234,0.15)]
                     inset-0 h-[100dvh] p-4 pb-[max(1rem,env(safe-area-inset-bottom))]
                     sm:inset-x-auto sm:inset-y-0 sm:right-0 sm:h-auto sm:w-96 sm:border-l sm:p-5"
        >
           <div className="flex justify-between items-center mb-4 sm:mb-6 shrink-0">
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                <h2 className="text-lg font-bold text-slate-100">Citizen AI</h2>
              </div>
              <button
                onClick={() => setShowAiDrawer(false)}
                aria-label="Close Citizen AI"
                className="h-9 w-9 flex items-center justify-center rounded-lg text-slate-400 hover:text-white hover:bg-slate-800"
              >✕</button>
           </div>
           
           <div className="flex-1 min-h-0 overflow-y-auto bg-slate-900/50 rounded-lg p-4 border border-slate-800 mb-4 text-sm text-slate-300">
             {aiResponse ? (
                <div className="whitespace-pre-wrap break-words">{aiResponse}</div>
             ) : (
                <div className="text-slate-500 italic">
                  Ask questions like:<br/><br/>
                  • "Why is there a traffic alert nearby?"<br/>
                  • "What active incidents should I avoid?"<br/>
                  • "Which emergency service is closest?"
                </div>
             )}
             {isAiLoading && <div className="mt-4 text-purple-400 animate-pulse">Analyzing platform data...</div>}
           </div>

           <div className="shrink-0 flex gap-2">
             {/* text-base on phones stops iOS Safari zooming into the field */}
             <input 
               type="text" 
               value={aiQuery}
               onChange={(e) => setAiQuery(e.target.value)}
               onKeyDown={(e) => e.key === 'Enter' && handleAskAI()}
               placeholder="Ask about your area..."
               className="flex-1 min-w-0 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2.5 text-base sm:text-sm text-white outline-none focus:border-purple-500"
             />
             <button onClick={handleAskAI} disabled={isAiLoading} aria-label="Send question" className="bg-purple-600 hover:bg-purple-500 disabled:opacity-60 text-white px-4 rounded-lg flex items-center justify-center transition-colors">
                ➤
             </button>
           </div>
        </div>
      )}

    </div>
  );
}

// Subcomponents
const KPICard = ({ title, count, color, className = '' }) => (
  <div className={`bg-slate-900/50 border border-slate-800 p-3 sm:p-4 rounded-xl flex flex-col items-center justify-center text-center ${className}`}>
    <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider mb-1">{title}</span>
    <span className={`text-2xl sm:text-3xl font-mono font-bold ${color}`}>{count}</span>
  </div>
);

const DetailRow = ({ label, value }) => (
  <div className="flex flex-col border-b border-slate-800 pb-2">
    <span className="text-[10px] uppercase text-slate-500 font-bold tracking-wider">{label}</span>
    <span className="text-sm text-slate-200 mt-0.5 break-words">{value}</span>
  </div>
);