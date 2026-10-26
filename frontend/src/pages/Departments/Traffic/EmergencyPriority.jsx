import React from 'react';

export default function EmergencyPriority() {
  return (
    <div className="flex flex-col gap-6 h-full">
      <header className="flex justify-between items-end">
        <div>
          <h1 className="text-2xl font-bold text-gray-100">Emergency Vehicle Priority</h1>
          <p className="text-sm text-gray-400 mt-1">Green corridor management and automated signal preemption</p>
        </div>
        <div className="text-sm bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 px-3 py-1 rounded shadow-[0_0_10px_rgba(16,185,129,0.2)]">
          1 Active Corridor
        </div>
      </header>

      <div className="flex-grow grid grid-cols-12 gap-6">
        
        {/* Active Corridors List */}
        <div className="col-span-4 flex flex-col gap-4">
          <div className="glass-panel p-4 border-l-4 border-emerald-500 bg-emerald-900/10 hover:bg-emerald-900/20 transition-colors cursor-pointer">
            <div className="flex justify-between items-start mb-2">
              <span className="font-bold text-gray-200">AMB-404 (City Hospital)</span>
              <span className="text-xs bg-emerald-500/20 text-emerald-300 px-2 py-1 rounded border border-emerald-500/30">En Route</span>
            </div>
            <div className="text-sm text-gray-400">Destination: MG Road Junction</div>
            <div className="text-sm text-gray-400 mt-1">ETA: 3 min 45 sec</div>
            <div className="mt-3 text-xs font-mono text-emerald-400">Next Signal Override: SIG-101</div>
          </div>

          <div className="glass-panel p-4 border-l-4 border-city-700 opacity-50">
            <div className="flex justify-between items-start mb-2">
              <span className="font-bold text-gray-200">FIRE-02 (Central Station)</span>
              <span className="text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded">Standby</span>
            </div>
            <div className="text-sm text-gray-400">No active dispatch</div>
          </div>
        </div>

        {/* Map visualization for the specific corridor */}
        <div className="col-span-8 glass-panel flex flex-col overflow-hidden border-emerald-500/30">
          <div className="bg-emerald-900/10 border-b border-emerald-500/30 p-3 text-sm font-bold uppercase tracking-wider text-emerald-300">
             Route Preemption Map
          </div>
          <div className="flex-grow flex items-center justify-center text-city-700 font-mono">
            [ Leaflet Map: Highlighting AMB-404 Route & Green Signals ]
          </div>
        </div>

      </div>
    </div>
  );
}