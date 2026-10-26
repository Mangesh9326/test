import React from 'react';
import DataTable from '../../../components/DataTable';
import KPICard from '../../../components/KPICard';

export default function Incidents() {
  const activeIncidents = [
    { id: 'INC-088', type: 'Severe Collision', location: 'MG Road Junction', severity: 'Critical', status: 'AI Responding' },
    { id: 'INC-089', type: 'Vehicle Breakdown', location: 'Station Road', severity: 'Low', status: 'Cleared' },
  ];

  return (
    <div className="flex flex-col gap-6 h-full">
      <header>
        <h1 className="text-2xl font-bold text-gray-100">Traffic Incidents</h1>
        <p className="text-sm text-gray-400 mt-1">Live accident tracking and cross-departmental response status</p>
      </header>

      <div className="grid grid-cols-3 gap-4">
        <KPICard title="Active Incidents" value="1" status="critical" />
        <KPICard title="Avg Response Time" value="4.2 min" status="nominal" />
        <KPICard title="Pending Clearances" value="1" status="warning" />
      </div>

      <div className="grid grid-cols-3 gap-6 flex-grow">
        {/* Incident List */}
        <div className="col-span-2 flex flex-col gap-4">
          <h3 className="font-bold text-gray-300 uppercase text-sm tracking-wider">Incident Log</h3>
          <DataTable 
            headers={['Incident ID', 'Type', 'Location', 'Severity', 'Status']}
            data={activeIncidents}
          />
        </div>

        {/* AI Action Plan Context Panel */}
        <div className="col-span-1 glass-panel flex flex-col border-indigo-500/30">
          <div className="bg-indigo-900/20 border-b border-indigo-500/30 p-3">
             <span className="text-sm font-bold uppercase tracking-wider text-indigo-300">Active AI Action Plan</span>
          </div>
          <div className="flex-grow p-4 overflow-y-auto space-y-4">
             <div className="bg-city-800 p-3 rounded border border-city-700">
               <div className="text-xs text-gray-400 mb-1">Target: INC-088 (MG Road)</div>
               <div className="text-sm text-gray-200 font-medium">Cross-Domain Actions Initiated:</div>
               <ul className="mt-2 text-xs text-gray-400 space-y-1">
                 <li className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span> Dispatched 2 Ambulances (Hospital Dept)</li>
                 <li className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span> Road Closure Active (Police Dept)</li>
                 <li className="flex items-center gap-2"><span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse"></span> Rerouting Traffic (Traffic Dept)</li>
               </ul>
             </div>
          </div>
        </div>
      </div>
    </div>
  );
}