import React from 'react';

const TrafficDashboard = () => {
  return (
    <div className="min-h-screen p-6 flex flex-col gap-6">
      
      <header>
        <h1 className="text-3xl font-bold text-gray-100">Traffic Operations</h1>
        <p className="text-sm text-gray-400 mt-1">Dedicated monitoring and signal control</p>
      </header>

      <div className="grid grid-cols-4 gap-4">
        {/* Expanded KPIs */}
        <div className="glass-panel p-4 border-l-4 border-l-blue-500">
          <div className="text-sm text-gray-400 uppercase">Active Vehicles</div>
          <div className="text-2xl font-bold text-gray-100 mt-1">12,450</div>
        </div>
        <div className="glass-panel p-4 border-l-4 border-l-red-500">
          <div className="text-sm text-gray-400 uppercase">Congestion Alerts</div>
          <div className="text-2xl font-bold text-gray-100 mt-1">3</div>
        </div>
        <div className="glass-panel p-4 border-l-4 border-l-emerald-500">
          <div className="text-sm text-gray-400 uppercase">Signal Efficiency</div>
          <div className="text-2xl font-bold text-gray-100 mt-1">94%</div>
        </div>
        <div className="glass-panel p-4 border-l-4 border-l-orange-500">
          <div className="text-sm text-gray-400 uppercase">Road Closures</div>
          <div className="text-2xl font-bold text-gray-100 mt-1">1</div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6 flex-grow">
        {/* Advanced Feature: Signal Control */}
        <div className="glass-panel col-span-1 p-4 flex flex-col">
          <h3 className="font-bold text-gray-300 mb-4 border-b border-city-700/50 pb-2">Traffic Signal Override</h3>
          <div className="flex-grow flex items-center justify-center text-sm text-gray-500">
            [Signal Control Interface Placeholder]
          </div>
        </div>

        {/* Advanced Feature: CCTV Feeds */}
        <div className="glass-panel col-span-2 p-4 flex flex-col">
          <h3 className="font-bold text-gray-300 mb-4 border-b border-city-700/50 pb-2">Live CCTV Analysis</h3>
          <div className="flex-grow grid grid-cols-2 gap-4">
            <div className="bg-black rounded-lg border border-city-700 flex items-center justify-center text-xs text-red-500 font-mono relative">
              <span className="absolute top-2 left-2 flex items-center gap-2">
                 <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span> LIVE REC 00:15
              </span>
              [YOLO Processed Feed - MG Road]
            </div>
            <div className="bg-black rounded-lg border border-city-700 flex items-center justify-center text-xs text-gray-600 font-mono">
              [Feed - Station Road]
            </div>
          </div>
        </div>
      </div>

    </div>
  );
};

export default TrafficDashboard;