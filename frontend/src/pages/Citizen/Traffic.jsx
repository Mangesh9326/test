import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useCitizenStore } from '../../store/useCitizenStore';

// --- ICONS ---
const AlertIcon = () => <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>;
const MapPinIcon = () => <svg className="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" /></svg>;
const NewsIcon = () => <svg className="w-5 h-5 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" /></svg>;
const BrainIcon = () => <svg className="w-5 h-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>;

export default function CitizenTraffic() {
  const { 
    location, 
    incidents, 
    traffic, 
    alerts, 
    isSimulated, 
    isLoading: storeLoading, 
    fetchAroundMe 
  } = useCitizenStore();

  const mapRef = useRef(null);
  const markersRef = useRef([]);
  
  const [mapObj, setMapObj] = useState(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [mapError, setMapError] = useState(null);
  
  const [activeFilter, setActiveFilter] = useState('All');
  const [selectedItem, setSelectedItem] = useState(null);
  const [news, setNews] = useState([]);
  const [isNewsLoading, setIsNewsLoading] = useState(false);
  
  const [aiExplanation, setAiExplanation] = useState('');
  const [isAiLoading, setIsAiLoading] = useState(false);

  // -----------------------------------------------------
  // 1. FETCH TRAFFIC NEWS (Backend Proxy)
  // -----------------------------------------------------
  useEffect(() => {
    const fetchNews = async () => {
      setIsNewsLoading(true);
      try {
        const res = await fetch(`/api/citizen/traffic/news?lat=${location.latitude}&lng=${location.longitude}`);
        if (res.ok) {
          const data = await res.json();
          setNews(data.news || []);
        } else {
          setNews([]);
        }
      } catch (err) {
        console.error("Traffic news is temporarily unavailable.", err);
      } finally {
        setIsNewsLoading(false);
      }
    };
    fetchNews();
  }, [location.latitude, location.longitude]);
// -----------------------------------------------------
  // 2. INITIALIZE GOOGLE MAPS
  // -----------------------------------------------------
  useEffect(() => {
    const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
    if (!apiKey) {
      setMapError("Live traffic map is unavailable. Configuration missing.");
      return;
    }

    const initMap = () => {
      if (mapRef.current && window.google && !mapObj) {
        const map = new window.google.maps.Map(mapRef.current, {
          center: { lat: parseFloat(location.latitude) || 19.0760, lng: parseFloat(location.longitude) || 72.8777 },
          zoom: 14,
          disableDefaultUI: true,
          zoomControl: true,
          mapTypeControl: false,
          styles: [
            { elementType: "geometry", stylers: [{ color: "#0f172a" }] },
            { elementType: "labels.text.stroke", stylers: [{ color: "#020617" }] },
            { elementType: "labels.text.fill", stylers: [{ color: "#94a3b8" }] },
            { featureType: "road", elementType: "geometry", stylers: [{ color: "#1e293b" }] }
          ]
        });

        const trafficLayer = new window.google.maps.TrafficLayer();
        trafficLayer.setMap(map);
        
        setMapObj(map);
        setMapLoaded(true);
      }
    };

    // If already loaded, initialize immediately
    if (window.google && window.google.maps) {
      initMap();
      return;
    }

    // Check for existing script to prevent React 18 Strict Mode duplicate injection
    const existingScript = document.getElementById('google-maps-script');
    
    if (!existingScript) {
      const script = document.createElement('script');
      script.id = 'google-maps-script';
      // Appended &loading=async to satisfy the new Google Maps requirement
      script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&loading=async`;
      script.async = true;
      script.defer = true;
      
      script.onload = initMap;
      script.onerror = () => setMapError("Live traffic map is unavailable.");
      
      document.head.appendChild(script);
    } else {
      // If script is in the DOM but still downloading, attach a secondary listener
      existingScript.addEventListener('load', initMap);
    }
  }, [mapObj, location.latitude, location.longitude]);

  // -----------------------------------------------------
  // 3. PLOT INCIDENT MARKERS
  // -----------------------------------------------------
  useEffect(() => {
    if (!mapObj || !window.google) return;

    // Clear old markers
    markersRef.current.forEach(m => m.setMap(null));
    markersRef.current = [];

    // Plot City Incidents over the Google Map
    incidents.forEach(inc => {
      if (inc.latitude && inc.longitude) {
        const marker = new window.google.maps.Marker({
          position: { lat: parseFloat(inc.latitude), lng: parseFloat(inc.longitude) },
          map: mapObj,
          icon: {
            path: window.google.maps.SymbolPath.CIRCLE,
            fillColor: inc.severity === 'CRITICAL' || inc.severity === 'HIGH' ? '#ef4444' : '#f59e0b',
            fillOpacity: 1,
            strokeWeight: 2,
            strokeColor: '#ffffff',
            scale: 8
          },
          title: inc.incident_type || "Incident"
        });

        marker.addListener('click', () => {
          setSelectedItem({ ...inc, itemType: 'incident' });
          setAiExplanation('');
        });
        
        markersRef.current.push(marker);
      }
    });
  }, [mapObj, incidents]);

  // -----------------------------------------------------
  // 4. EXPLAINABLE AI INTEGRATION
  // -----------------------------------------------------
  const handleAskAI = async (item) => {
    setIsAiLoading(true);
    setAiExplanation('');
    
    // Send structured evidence to the backend AI proxy
    const context = {
      message: `Why is traffic slow near ${item.location || item.name}?`,
      location: { latitude: location.latitude, longitude: location.longitude },
      selectedItem: item,
      trafficIncidents: incidents,
      alerts: alerts,
      trafficNews: news
    };

    try {
      const res = await fetch('/api/citizen/assistant', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: context.message, context })
      });
      
      if (!res.ok) throw new Error();
      const data = await res.json();
      setAiExplanation(data.reply);
    } catch (err) {
      setAiExplanation("AI explanation is temporarily unavailable.");
    } finally {
      setIsAiLoading(false);
    }
  };

  // -----------------------------------------------------
  // DATA FILTERING
  // -----------------------------------------------------
  const filteredList = useMemo(() => {
    let list = [];
    if (activeFilter === 'All' || activeFilter === 'Incidents') {
      list = [...list, ...incidents.map(i => ({ ...i, itemType: 'incident' }))];
    }
    if (activeFilter === 'All' || activeFilter === 'News') {
      list = [...list, ...news.map(n => ({ ...n, itemType: 'news' }))];
    }
    return list;
  }, [incidents, news, activeFilter]);

  return (
    <div className="flex flex-col min-h-full bg-slate-950 font-sans text-slate-200 w-full max-w-full">
      
      {/* HEADER */}
      <div className="flex justify-between items-center p-4 border-b border-slate-800 shrink-0">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-100 tracking-tight">Traffic Around You</h1>
          <p className="text-xs text-slate-400 mt-1 truncate flex items-center gap-1">
            <MapPinIcon /> {location.name || 'Current Location'}
          </p>
        </div>
        <button 
          onClick={fetchAroundMe}
          disabled={storeLoading}
          className="min-w-[44px] min-h-[44px] flex items-center justify-center bg-slate-900 hover:bg-slate-800 border border-slate-700 text-blue-400 rounded-full transition-colors shrink-0"
        >
          <svg className={`w-5 h-5 ${storeLoading ? 'animate-spin' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </button>
      </div>

      {/* RESPONSIVE LAYOUT SHELL */}
      <div className="flex-1 lg:grid lg:grid-cols-12 lg:gap-6 lg:p-6 lg:items-start overflow-y-auto custom-scrollbar pb-[env(safe-area-inset-bottom)]">
        
        {/* LEFT/TOP: Google Map & Summary */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          
          <div className="relative w-full h-[320px] sm:h-[400px] lg:h-[500px] lg:rounded-2xl overflow-hidden shadow-lg lg:border border-slate-800 z-0 shrink-0 bg-slate-900">
            {/* Source Label Overlay */}
            <div className="absolute top-3 right-3 z-10 bg-black/60 backdrop-blur-sm border border-slate-700 px-2.5 py-1 rounded shadow text-[10px] font-bold text-slate-300 uppercase tracking-widest">
              Google Live Traffic
            </div>

            {mapError ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center text-slate-400">
                <AlertIcon />
                <span className="text-sm mt-2">{mapError}</span>
                <button onClick={() => window.location.reload()} className="mt-4 px-4 py-2 bg-slate-800 rounded-lg text-xs font-bold text-slate-200 min-h-[44px]">Retry</button>
              </div>
            ) : !mapLoaded ? (
              <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-sm">
                Loading Google Maps...
              </div>
            ) : null}
            <div ref={mapRef} className="absolute inset-0 w-full h-full" />
          </div>

          {/* TRAFFIC SUMMARY CARD */}
          <div className="mx-4 lg:mx-0 bg-slate-900 border border-slate-800 rounded-xl p-4 shadow flex flex-col">
            <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-1">Current Traffic</span>
            <div className="flex items-center justify-between">
              <span className={`text-xl font-bold ${incidents.length > 2 ? 'text-orange-400' : 'text-emerald-400'}`}>
                {incidents.length > 2 ? 'Heavy Congestion' : 'Moderate Flow'}
              </span>
              <div className="flex gap-4">
                <div className="text-center">
                  <span className="block text-lg font-mono font-bold text-slate-200">{incidents.length}</span>
                  <span className="text-[10px] text-slate-500 uppercase">Incidents</span>
                </div>
                {alerts.length > 0 && (
                  <div className="text-center">
                    <span className="block text-lg font-mono font-bold text-red-400">{alerts.length}</span>
                    <span className="text-[10px] text-slate-500 uppercase">Alerts</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT/BOTTOM: Feeds & AI Drawer */}
        <div className="lg:col-span-5 flex flex-col gap-4 mt-4 lg:mt-0 p-4 lg:p-0">
          
          {/* HORIZONTAL FILTERS (Safe scrolling) */}
          <div className="flex gap-2 overflow-x-auto custom-scrollbar pb-2 shrink-0 -mx-4 px-4 lg:mx-0 lg:px-0">
            {['All', 'Incidents', 'News'].map(filter => (
              <button 
                key={filter}
                onClick={() => setActiveFilter(filter)}
                className={`min-h-[44px] px-5 rounded-full text-sm font-bold transition-colors whitespace-nowrap shrink-0 border ${
                  activeFilter === filter 
                    ? 'bg-slate-200 text-slate-900 border-slate-200' 
                    : 'bg-slate-900 text-slate-400 border-slate-700 hover:bg-slate-800'
                }`}
              >
                {filter}
              </button>
            ))}
          </div>

          {/* LIST VIEWS */}
          <div className="flex flex-col gap-3">
            {filteredList.length === 0 ? (
               <div className="text-center py-8 border border-slate-800 border-dashed rounded-xl bg-slate-900/50">
                 <p className="text-slate-500 text-sm">No recent incidents or news reported.</p>
               </div>
            ) : (
               filteredList.map((item, i) => (
                 <button 
                   key={i}
                   onClick={() => { setSelectedItem(item); setAiExplanation(''); }}
                   className="text-left bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col gap-2 hover:border-slate-600 transition-colors focus:outline-none min-h-[44px]"
                 >
                   <div className="flex justify-between items-start w-full">
                     <h3 className="font-bold text-slate-200 text-sm flex-1 pr-2">
                       {item.itemType === 'incident' ? item.incident_type : item.headline}
                     </h3>
                     {item.itemType === 'incident' ? (
                       <span className={`text-[10px] px-2 py-0.5 rounded font-bold uppercase tracking-widest shrink-0 ${
                         ['CRITICAL', 'SEVERE', 'HIGH'].includes(String(item.severity).toUpperCase()) 
                           ? 'bg-red-500/20 text-red-400' 
                           : 'bg-orange-500/20 text-orange-400'
                       }`}>
                         {item.severity}
                       </span>
                     ) : (
                       <span className="text-[10px] bg-indigo-500/20 text-indigo-400 px-2 py-0.5 rounded font-bold uppercase tracking-widest shrink-0 flex items-center gap-1">
                         <NewsIcon/> News
                       </span>
                     )}
                   </div>
                   
                   <p className="text-xs text-slate-400 line-clamp-2">
                     {item.itemType === 'incident' ? item.location : item.summary}
                   </p>
                   
                   <div className="text-[10px] text-slate-500 mt-1 flex justify-between w-full">
                     {item.itemType === 'incident' ? (
                       <span>{Number(item.distance || 0).toFixed(1)} km away • {item.status}</span>
                     ) : (
                       <span>Source: {item.source} • {item.published_time || 'Recent'}</span>
                     )}
                   </div>
                 </button>
               ))
            )}
          </div>
        </div>
      </div>

      {/* MOBILE BOTTOM SHEET / DESKTOP MODAL */}
      {selectedItem && (
        <>
          <div 
            className="fixed inset-0 bg-black/70 z-[100] backdrop-blur-sm transition-opacity"
            onClick={() => setSelectedItem(null)}
          ></div>
          
          <div className="fixed bottom-0 left-0 right-0 lg:left-1/2 lg:top-1/2 lg:bottom-auto lg:-translate-x-1/2 lg:-translate-y-1/2 lg:w-full lg:max-w-lg bg-slate-900 border border-slate-700 lg:rounded-2xl rounded-t-3xl z-[101] flex flex-col max-h-[85vh] shadow-2xl">
            
            {/* Mobile Drag Handle */}
            <div className="w-full flex justify-center pt-3 pb-2 lg:hidden shrink-0" onClick={() => setSelectedItem(null)}>
              <div className="w-12 h-1.5 bg-slate-700 rounded-full"></div>
            </div>

            <div className="p-5 overflow-y-auto custom-scrollbar pb-[env(safe-area-inset-bottom)]">
              
              {/* Type Badge */}
              <div className="mb-2">
                <span className={`text-[10px] px-2 py-1 rounded font-bold uppercase tracking-widest ${selectedItem.itemType === 'incident' ? 'bg-red-500/20 text-red-400' : 'bg-indigo-500/20 text-indigo-400'}`}>
                  {selectedItem.itemType === 'incident' ? 'Traffic Incident' : 'Traffic News'}
                </span>
              </div>

              <h2 className="text-lg sm:text-xl font-bold text-slate-100 mb-3">
                {selectedItem.itemType === 'incident' ? selectedItem.incident_type : selectedItem.headline}
              </h2>
              
              {/* Details Grid */}
              <div className="grid grid-cols-2 gap-3 mb-6 bg-slate-950 p-4 rounded-xl border border-slate-800">
                <div className="col-span-2">
                  <span className="block text-[10px] text-slate-500 uppercase tracking-widest mb-1">Location / Impact</span>
                  <span className="font-medium text-sm text-slate-300">
                    {selectedItem.location || selectedItem.summary || "Location data unavailable"}
                  </span>
                </div>
                {selectedItem.itemType === 'incident' && (
                  <>
                    <div>
                      <span className="block text-[10px] text-slate-500 uppercase tracking-widest mb-1">Status</span>
                      <span className="font-bold text-sm text-slate-200">{selectedItem.status}</span>
                    </div>
                    <div>
                      <span className="block text-[10px] text-slate-500 uppercase tracking-widest mb-1">Distance</span>
                      <span className="font-bold text-sm text-slate-200">{Number(selectedItem.distance || 0).toFixed(1)} km</span>
                    </div>
                  </>
                )}
                {selectedItem.itemType === 'news' && (
                  <div className="col-span-2 flex justify-between">
                    <span className="text-xs text-slate-400">Source: {selectedItem.source}</span>
                    <span className="text-xs text-slate-400">{selectedItem.published_time}</span>
                  </div>
                )}
              </div>

              {/* Explainable AI Integration */}
              <div className="border-t border-slate-800 pt-5">
                {!aiExplanation ? (
                  <button 
                    onClick={() => handleAskAI(selectedItem)}
                    disabled={isAiLoading}
                    className="w-full min-h-[48px] bg-slate-800 hover:bg-slate-700 active:bg-slate-600 text-slate-200 border border-slate-700 font-bold rounded-xl flex items-center justify-center gap-2 transition-colors disabled:opacity-70"
                  >
                    <BrainIcon />
                    {isAiLoading ? 'Analyzing Context...' : 'Explain this with AI'}
                  </button>
                ) : (
                  <div className="p-4 bg-emerald-950/20 border border-emerald-900/50 rounded-xl animate-in fade-in slide-in-from-bottom-2">
                    <span className="block text-[10px] font-bold text-emerald-400 uppercase tracking-widest mb-2 flex items-center gap-2">
                      <BrainIcon /> AI Explanation
                    </span>
                    <p className="text-sm text-slate-200 leading-relaxed">{aiExplanation}</p>
                    
                    <div className="mt-4 pt-3 border-t border-emerald-900/30">
                      <span className="text-[9px] text-slate-500 uppercase">
                        Based on: ✓ Smart City incident data ✓ Traffic news ✓ Google Maps context
                      </span>
                      <p className="text-[10px] text-slate-500 italic mt-1">
                        AI-generated explanation based on available city and traffic data.
                      </p>
                    </div>
                  </div>
                )}
              </div>

            </div>
          </div>
        </>
      )}
    </div>
  );
}