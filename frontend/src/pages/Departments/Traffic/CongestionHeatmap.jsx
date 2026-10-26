import React from 'react';

export default function CongestionHeatmap() {
  return (
    <div className="flex flex-col gap-4 h-full relative">
      <header className="absolute top-4 left-4 z-20 glass-panel p-4 w-80">
        <h1 className="text-xl font-bold text-gray-100">Congestion Heatmap</h1>
        <p className="text-xs text-gray-400 mt-1 mb-4">Historical vs Live density overlay</p>
        
        <div className="flex flex-col gap-2">
          <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
            <input type="checkbox" defaultChecked className="accent-blue-500" />
            Live Traffic Density
          </label>
          <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
            <input type="checkbox" className="accent-blue-500" />
            Historical Average (Tuesday 10:00 AM)
          </label>
          <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
            <input type="checkbox" defaultChecked className="accent-blue-500" />
            AI Bottleneck Predictions
          </label>
        </div>
      </header>

      {/* Full Area Map Overlay */}
      <div className="flex-grow glass-panel border-city-700/50 flex items-center justify-center text-city-700 font-mono">
        [ React Leaflet Heatmap Layer Injection Point ]
      </div>
    </div>
  );
}