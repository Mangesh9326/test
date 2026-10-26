import React, { useEffect, useState, useMemo } from 'react';
import { useAlertsStore } from '../../store/useAlertsStore';

// --- Reusable SVG Icons ---
const AlertIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
);
const RefreshIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
);
const CloseIcon = ({ className = "w-6 h-6" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
);

export default function Alerts() {
  const { alerts, readAlertIds, isLoading, error, fetchAlerts, subscribeToSockets, unsubscribeFromSockets, markAsRead, askAI } = useAlertsStore();
  const [activeFilter, setActiveFilter] = useState('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [aiExplanation, setAiExplanation] = useState(null);
  const [isAiLoading, setIsAiLoading] = useState(false);

  useEffect(() => {
    fetchAlerts();
    subscribeToSockets();
    return () => unsubscribeFromSockets();
  }, [fetchAlerts, subscribeToSockets, unsubscribeFromSockets]);

  // Derived Statistics
  const criticalCount = alerts.filter(a => a.severity === 'CRITICAL').length;
  const activeCount = alerts.length;
  const unreadCount = alerts.filter(a => !readAlertIds.has(a.alert_id)).length;
  const nearbyCount = alerts.filter(a => a.distance <= 5).length;

  // Filtered & Sorted Alerts
  const displayedAlerts = useMemo(() => {
    let filtered = alerts;
    if (activeFilter !== 'ALL') {
      filtered = filtered.filter(a => a.category === activeFilter);
    }
    if (searchQuery.trim() !== '') {
      const q = searchQuery.toLowerCase();
      filtered = filtered.filter(a => a.title.toLowerCase().includes(q) || a.location.toLowerCase().includes(q) || a.message.toLowerCase().includes(q));
    }
    return filtered;
  }, [alerts, activeFilter, searchQuery]);

  const categories = ['ALL', ...Array.from(new Set(alerts.map(a => a.category)))];

  const handleOpenAlert = (alert) => {
    setSelectedAlert(alert);
    setAiExplanation(null);
    markAsRead(alert.alert_id);
  };

  const handleAskAI = async () => {
    if (!selectedAlert) return;
    setIsAiLoading(true);
    const explanation = await askAI("Why am I seeing this alert? Explain the cause and impact.", selectedAlert);
    setAiExplanation(explanation);
    setIsAiLoading(false);
  };

  const getSeverityStyles = (severity) => {
    switch(severity) {
      case 'CRITICAL': return 'bg-red-100 text-red-800 border-red-300 dark:bg-red-900/30 dark:text-red-400 dark:border-red-800';
      case 'HIGH': return 'bg-orange-100 text-orange-800 border-orange-300 dark:bg-orange-900/30 dark:text-orange-400 dark:border-orange-800';
      case 'MODERATE':
      case 'MEDIUM': return 'bg-yellow-100 text-yellow-800 border-yellow-300 dark:bg-yellow-900/30 dark:text-yellow-400 dark:border-yellow-800';
      default: return 'bg-blue-100 text-blue-800 border-blue-300 dark:bg-blue-900/30 dark:text-blue-400 dark:border-blue-800';
    }
  };

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center p-8 text-center">
        <p className="text-gray-600 dark:text-gray-400 mb-4">Unable to load alerts.</p>
        <button onClick={fetchAlerts} className="min-h-[44px] px-6 bg-blue-600 text-white rounded-lg font-medium active:bg-blue-700 transition-colors">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col w-full h-full pb-20 overflow-y-auto custom-scrollbar bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      
      {/* HEADER */}
      <header className="sticky top-0 z-10 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border-b border-gray-200 dark:border-gray-800 p-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">Alerts</h1>
            <p className="text-xs text-gray-500 dark:text-gray-400">Important updates and warnings around you.</p>
          </div>
          <button 
            onClick={fetchAlerts} 
            className="p-3 bg-gray-100 dark:bg-gray-800 rounded-full active:scale-95 transition-transform min-w-[44px] min-h-[44px] flex items-center justify-center"
            aria-label="Refresh alerts"
          >
            <RefreshIcon className={`w-5 h-5 ${isLoading ? 'animate-spin text-blue-500' : 'text-gray-600 dark:text-gray-300'}`} />
          </button>
        </div>
      </header>

      <div className="p-4 space-y-4 max-w-7xl mx-auto w-full">
        
        {/* SUMMARY CARDS */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="bg-white dark:bg-gray-800 p-3 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700">
            <div className="text-xs text-gray-500 dark:text-gray-400 uppercase font-semibold">Critical</div>
            <div className="text-2xl font-bold text-red-600 dark:text-red-400">{criticalCount}</div>
          </div>
          <div className="bg-white dark:bg-gray-800 p-3 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700">
            <div className="text-xs text-gray-500 dark:text-gray-400 uppercase font-semibold">Active</div>
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{activeCount}</div>
          </div>
          <div className="bg-white dark:bg-gray-800 p-3 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700">
            <div className="text-xs text-gray-500 dark:text-gray-400 uppercase font-semibold">Unread</div>
            <div className="text-2xl font-bold text-gray-800 dark:text-gray-200">{unreadCount}</div>
          </div>
          <div className="bg-white dark:bg-gray-800 p-3 rounded-xl shadow-sm border border-gray-100 dark:border-gray-700">
            <div className="text-xs text-gray-500 dark:text-gray-400 uppercase font-semibold">Nearby</div>
            <div className="text-2xl font-bold text-teal-600 dark:text-teal-400">{nearbyCount}</div>
          </div>
        </div>

        {/* SEARCH & FILTERS */}
        <div className="space-y-3">
          <input 
            type="text" 
            placeholder="Search alerts..." 
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full min-h-[44px] px-4 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          
          <div className="flex gap-2 overflow-x-auto pb-2 custom-scrollbar snap-x">
            {categories.map(cat => (
              <button
                key={cat}
                onClick={() => setActiveFilter(cat)}
                className={`snap-start whitespace-nowrap px-4 py-2 min-h-[44px] rounded-full text-sm font-medium transition-colors border ${
                  activeFilter === cat 
                    ? 'bg-blue-600 text-white border-blue-600' 
                    : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border-gray-200 dark:border-gray-700'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        {/* ALERT LIST */}
        <div className="space-y-3">
          {isLoading ? (
            // Skeleton Loader
            [1,2,3].map(i => (
              <div key={i} className="bg-white dark:bg-gray-800 p-4 rounded-xl border border-gray-100 dark:border-gray-700 animate-pulse">
                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/4 mb-3"></div>
                <div className="h-5 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-2"></div>
                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-4"></div>
                <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-full"></div>
              </div>
            ))
          ) : displayedAlerts.length === 0 ? (
            <div className="text-center py-10 text-gray-500">
              {searchQuery ? "No alerts match your search." : "No active alerts."}
            </div>
          ) : (
            displayedAlerts.map(alert => {
              const isUnread = !readAlertIds.has(alert.alert_id);
              return (
                <div 
                  key={alert.alert_id} 
                  className={`bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border transition-all ${
                    alert.severity === 'CRITICAL' ? 'border-red-400 dark:border-red-600/50' : 'border-gray-200 dark:border-gray-700'
                  } ${isUnread ? 'ring-1 ring-blue-500/50' : ''}`}
                >
                  <div className="flex justify-between items-start mb-2">
                    <span className={`px-2 py-1 rounded text-[10px] font-bold tracking-wider border ${getSeverityStyles(alert.severity)}`}>
                      {alert.severity === 'CRITICAL' ? '🚨 ' : ''}{alert.severity}
                    </span>
                    {isUnread && <span className="w-2.5 h-2.5 bg-blue-500 rounded-full mt-1"></span>}
                  </div>
                  
                  <h3 className="text-lg font-bold mb-1 leading-tight text-gray-900 dark:text-white">
                    {alert.title}
                  </h3>
                  
                  <p className="text-sm text-gray-600 dark:text-gray-300 line-clamp-2 mb-3">
                    {alert.message}
                  </p>
                  
                  <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-500 dark:text-gray-400 font-medium mb-4">
                    <span className="flex items-center gap-1">📍 {alert.distance} km away</span>
                    <span className="flex items-center gap-1">🕐 {new Date(alert.created_at).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                  </div>

                  <div className="flex flex-col sm:flex-row gap-2 justify-between items-center border-t border-gray-100 dark:border-gray-700 pt-3">
                    {alert.is_simulated && (
                      <span className="text-[10px] font-mono bg-gray-100 dark:bg-gray-700 text-gray-500 px-2 py-1 rounded">
                        SIMULATED DATA
                      </span>
                    )}
                    <button 
                      onClick={() => handleOpenAlert(alert)}
                      className="w-full sm:w-auto min-h-[44px] px-4 bg-gray-100 hover:bg-gray-200 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-900 dark:text-white text-sm font-semibold rounded-lg transition-colors"
                    >
                      View Details
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* ALERT DETAILS DRAWER (Bottom Sheet for Mobile / Modal for Desktop) */}
      {selectedAlert && (
        <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-black/50 backdrop-blur-sm p-0 sm:p-4 animate-in fade-in">
          <div className="w-full sm:max-w-md bg-white dark:bg-gray-900 rounded-t-2xl sm:rounded-2xl overflow-hidden shadow-2xl animate-in slide-in-from-bottom-8">
            
            <div className="flex justify-between items-center p-4 border-b border-gray-200 dark:border-gray-800">
              <h2 className="font-bold text-lg">{selectedAlert.category} Alert</h2>
              <button 
                onClick={() => setSelectedAlert(null)}
                className="p-2 -mr-2 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 rounded-full min-w-[44px] min-h-[44px] flex items-center justify-center transition-colors"
              >
                <CloseIcon />
              </button>
            </div>

            <div className="p-5 overflow-y-auto max-h-[70vh] custom-scrollbar">
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">{selectedAlert.title}</h3>
              
              <div className="grid grid-cols-2 gap-y-3 text-sm mb-6 bg-gray-50 dark:bg-gray-800/50 p-4 rounded-xl">
                <div><span className="text-gray-500 block text-xs">Severity</span><span className={`font-semibold ${selectedAlert.severity === 'CRITICAL' ? 'text-red-500' : ''}`}>{selectedAlert.severity}</span></div>
                <div><span className="text-gray-500 block text-xs">Status</span><span className="font-semibold">{selectedAlert.status}</span></div>
                <div className="col-span-2"><span className="text-gray-500 block text-xs">Location</span><span className="font-semibold">{selectedAlert.location}</span></div>
                <div><span className="text-gray-500 block text-xs">Distance</span><span className="font-semibold">{selectedAlert.distance} km</span></div>
                <div><span className="text-gray-500 block text-xs">Source</span><span className="font-semibold">{selectedAlert.source}</span></div>
              </div>

              <div className="mb-6">
                <span className="text-gray-500 block text-xs mb-1">Details</span>
                <p className="text-sm text-gray-800 dark:text-gray-200 leading-relaxed">
                  {selectedAlert.message}
                </p>
              </div>

              {/* AI EXPLANATION SECTION */}
              <div className="mb-6 border-t border-gray-100 dark:border-gray-800 pt-4">
                {!aiExplanation ? (
                  <button 
                    onClick={handleAskAI}
                    disabled={isAiLoading}
                    className="w-full min-h-[44px] bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-semibold rounded-xl flex items-center justify-center gap-2 transition-all disabled:opacity-70"
                  >
                    {isAiLoading ? (
                      <RefreshIcon className="animate-spin w-5 h-5" />
                    ) : (
                      <>✨ Explain with AI</>
                    )}
                  </button>
                ) : (
                  <div className="bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-100 dark:border-indigo-800/50 p-4 rounded-xl">
                    <h4 className="flex items-center gap-2 text-indigo-700 dark:text-indigo-400 font-bold text-sm mb-2">
                      ✨ AI-generated explanation
                    </h4>
                    <p className="text-sm text-indigo-900 dark:text-indigo-200 leading-relaxed mb-3">
                      {aiExplanation}
                    </p>
                    <div className="text-[10px] text-indigo-500 dark:text-indigo-400/70 border-t border-indigo-200/50 dark:border-indigo-800/50 pt-2">
                      Based on: ✓ Smart City incident data ✓ Alert context
                    </div>
                  </div>
                )}
              </div>

              <div className="flex flex-col gap-2">
                <button className="w-full min-h-[44px] bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-900 dark:text-white font-semibold rounded-xl transition-colors">
                  View on Map
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}