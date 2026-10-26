import React, { useEffect, useState } from 'react';
import { useServicesStore } from '../../store/useServicesStore';
import { Search, MapPin, Phone, Shield, Flame, Activity, Info, X, Map, Sparkles } from 'lucide-react';

const CATEGORIES = ['All', 'Emergency', 'Healthcare', 'Police', 'Fire', 'Traffic', 'Utilities', 'Transport', 'Civic', 'Other'];

export default function CitizenServices() {
  const { services, isLoading, error, isSimulated, searchQuery, selectedCategory, setSearchQuery, setSelectedCategory, fetchServices, subscribeToSockets, unsubscribeFromSockets, askAI } = useServicesStore();
  const [selectedService, setSelectedService] = useState(null);
  const [aiExplanation, setAiExplanation] = useState(null);
  const [isAiLoading, setIsAiLoading] = useState(false);

  useEffect(() => {
    fetchServices();
    subscribeToSockets();
    return () => unsubscribeFromSockets();
  }, [fetchServices, subscribeToSockets, unsubscribeFromSockets]);

  const handleExplainAI = async (service) => {
    setIsAiLoading(true);
    setAiExplanation(null);
    const explanation = await askAI("What is this service and when should I use it?", service);
    setAiExplanation(explanation);
    setIsAiLoading(false);
  };

  const getStatusBadge = (status) => {
    const s = (status || 'UNKNOWN').toUpperCase();
    if (s.includes('AVAILABLE') || s.includes('ACTIVE')) return <span className="bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded text-[10px] font-bold uppercase border border-emerald-500/30">Available</span>;
    if (s.includes('BUSY') || s.includes('LIMITED')) return <span className="bg-amber-500/20 text-amber-400 px-2 py-0.5 rounded text-[10px] font-bold uppercase border border-amber-500/30">Busy</span>;
    if (s.includes('UNAVAILABLE') || s.includes('CLOSED')) return <span className="bg-red-500/20 text-red-400 px-2 py-0.5 rounded text-[10px] font-bold uppercase border border-red-500/30">Closed</span>;
    return <span className="bg-slate-700 text-slate-300 px-2 py-0.5 rounded text-[10px] font-bold uppercase border border-slate-600">Unknown</span>;
  };

  return (
    <div className="flex flex-col min-h-screen bg-slate-950 pb-24">
      {/* HEADER */}
      <div className="pt-4 px-4 pb-2 sticky top-0 z-40 bg-slate-950/90 backdrop-blur-md">
        <h1 className="text-2xl font-black text-white">City Services</h1>
        <p className="text-sm text-slate-400">Find essential services around you</p>
        
        {isSimulated && (
          <div className="mt-2 bg-blue-500/10 border border-blue-500/30 text-blue-400 text-[10px] font-bold px-2 py-1 rounded inline-block uppercase tracking-wider">
            Simulated Data
          </div>
        )}

        {/* SEARCH */}
        <div className="mt-4 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            type="text"
            placeholder="Search services..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 rounded-xl py-3 pl-10 pr-10 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 transition-colors"
          />
          {searchQuery && (
            <button onClick={() => setSearchQuery('')} className="absolute right-3 top-1/2 -translate-y-1/2 p-1 text-slate-400 hover:text-white min-h-[44px] min-w-[44px] flex items-center justify-center">
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* CATEGORY CHIPS */}
        <div className="flex overflow-x-auto gap-2 mt-4 pb-2 scrollbar-hide -mx-4 px-4">
          {CATEGORIES.map(cat => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`whitespace-nowrap px-4 py-2 rounded-full text-sm font-semibold transition-colors min-h-[44px] ${
                selectedCategory === cat ? 'bg-blue-600 text-white' : 'bg-slate-900 text-slate-400 border border-slate-800'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      {/* CONTENT */}
      <div className="px-4 mt-2 flex flex-col gap-3">
        {/* EMERGENCY QUICK LINKS (Only show when "All" or "Emergency" is selected and no search) */}
        {!searchQuery && (selectedCategory === 'All' || selectedCategory === 'Emergency') && (
          <div className="grid grid-cols-2 gap-3 mb-2">
            <div className="bg-red-500/10 border border-red-500/30 p-3 rounded-2xl flex items-center gap-3 active:bg-red-500/20">
              <div className="bg-red-500/20 p-2 rounded-full"><Flame className="w-5 h-5 text-red-400" /></div>
              <div><h3 className="text-red-400 font-bold text-sm">Fire</h3><p className="text-[10px] text-slate-400">Emergency</p></div>
            </div>
            <div className="bg-blue-500/10 border border-blue-500/30 p-3 rounded-2xl flex items-center gap-3 active:bg-blue-500/20">
              <div className="bg-blue-500/20 p-2 rounded-full"><Shield className="w-5 h-5 text-blue-400" /></div>
              <div><h3 className="text-blue-400 font-bold text-sm">Police</h3><p className="text-[10px] text-slate-400">Security</p></div>
            </div>
            <div className="bg-emerald-500/10 border border-emerald-500/30 p-3 rounded-2xl flex items-center gap-3 active:bg-emerald-500/20">
              <div className="bg-emerald-500/20 p-2 rounded-full"><Activity className="w-5 h-5 text-emerald-400" /></div>
              <div><h3 className="text-emerald-400 font-bold text-sm">Medical</h3><p className="text-[10px] text-slate-400">Ambulance</p></div>
            </div>
          </div>
        )}

        <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mt-2 mb-1">Nearby Services</h2>
        
        {/* STATES */}
        {isLoading ? (
          Array(4).fill(0).map((_, i) => (
            <div key={i} className="bg-slate-900 h-24 rounded-2xl animate-pulse"></div>
          ))
        ) : error ? (
          <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl text-center">
            <p className="text-slate-400 text-sm mb-3">{error}</p>
            <button onClick={fetchServices} className="bg-slate-800 text-white px-4 py-2 rounded-lg text-sm font-semibold min-h-[44px]">Retry</button>
          </div>
        ) : services.length === 0 ? (
          <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl text-center">
            <p className="text-slate-400 text-sm">No services match your search or category.</p>
          </div>
        ) : (
          services.map(service => (
            <div 
              key={service.service_id} 
              onClick={() => { setSelectedService(service); setAiExplanation(null); }}
              className="bg-slate-900 border border-slate-800 p-4 rounded-2xl active:bg-slate-800 transition-colors cursor-pointer"
            >
              <div className="flex justify-between items-start mb-2">
                <h3 className="font-bold text-slate-200 text-base leading-tight">{service.name}</h3>
                {getStatusBadge(service.status)}
              </div>
              <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-1">
                <MapPin className="w-3 h-3" />
                <span>{service.area} • {service.distance_km} km</span>
              </div>
              <div className="flex gap-2 mt-3">
                <span className="text-[10px] bg-slate-800 text-slate-300 px-2 py-1 rounded font-semibold">{service.category}</span>
                {service.emergency_available && <span className="text-[10px] bg-red-500/20 text-red-400 border border-red-500/20 px-2 py-1 rounded font-semibold flex items-center gap-1"><Activity className="w-3 h-3" /> ER Available</span>}
              </div>
            </div>
          ))
        )}
      </div>

      {/* SERVICE DETAIL BOTTOM SHEET */}
      {selectedService && (
        <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/60 backdrop-blur-sm p-4 sm:p-6" onClick={() => setSelectedService(null)}>
          <div 
            className="bg-slate-900 w-full max-w-lg rounded-3xl border border-slate-800 overflow-hidden flex flex-col max-h-[85vh]"
            onClick={e => e.stopPropagation()}
            style={{ paddingBottom: 'env(safe-area-inset-bottom)' }}
          >
            <div className="p-5 border-b border-slate-800 relative">
              <button onClick={() => setSelectedService(null)} className="absolute right-4 top-4 text-slate-400 hover:text-white bg-slate-800 rounded-full p-1.5 min-h-[44px] min-w-[44px] flex items-center justify-center">
                <X className="w-5 h-5" />
              </button>
              <div className="mb-2">{getStatusBadge(selectedService.status)}</div>
              <h2 className="text-xl font-bold text-white mb-1 pr-10">{selectedService.name}</h2>
              <p className="text-sm text-slate-400 flex items-center gap-1.5">
                <MapPin className="w-4 h-4" /> {selectedService.address} ({selectedService.distance_km} km)
              </p>
            </div>

            <div className="p-5 overflow-y-auto">
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 mb-4">
                <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">Description</h3>
                <p className="text-sm text-slate-300">{selectedService.description}</p>
                <div className="mt-3 flex gap-4 text-sm">
                  <div>
                    <span className="text-slate-500 text-xs block mb-0.5">Department</span>
                    <span className="text-slate-300 font-semibold">{selectedService.department}</span>
                  </div>
                  {selectedService.emergency_available && (
                    <div>
                      <span className="text-slate-500 text-xs block mb-0.5">Emergency</span>
                      <span className="text-red-400 font-semibold">24/7 Available</span>
                    </div>
                  )}
                </div>
              </div>

              {/* AI EXPLANATION SECTION */}
              <div className="bg-blue-900/10 border border-blue-500/20 p-4 rounded-xl mb-4">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-bold text-blue-400 flex items-center gap-1.5"><Sparkles className="w-4 h-4" /> AI Assistant</h3>
                </div>
                {aiExplanation ? (
                  <p className="text-sm text-slate-300 leading-relaxed">{aiExplanation}</p>
                ) : (
                  <button 
                    onClick={() => handleExplainAI(selectedService)}
                    disabled={isAiLoading}
                    className="w-full bg-blue-600/20 text-blue-400 py-3 rounded-xl text-sm font-bold active:bg-blue-600/30 transition-colors min-h-[44px] flex justify-center items-center gap-2"
                  >
                    {isAiLoading ? 'Analyzing Data...' : 'Explain this service with AI'}
                  </button>
                )}
              </div>

              {/* ACTIONS */}
              <div className="grid grid-cols-2 gap-3 mt-4">
                <button className="bg-slate-800 text-white font-semibold py-3 rounded-xl text-sm min-h-[44px] flex items-center justify-center gap-2 active:bg-slate-700">

<Map className="w-4 h-4" />
View on Map
                </button>
                <button className="bg-blue-600 text-white font-semibold py-3 rounded-xl text-sm min-h-[44px] flex items-center justify-center gap-2 active:bg-blue-500">
                  <Phone className="w-4 h-4" /> Call Dept
                </button>
              </div>
              
              {selectedService.is_simulated && (
                <p className="text-center text-[10px] text-slate-500 uppercase tracking-widest mt-4">
                  Smart City Platform • Simulated Data
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}