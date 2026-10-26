import React from 'react';
import { useNavigate } from 'react-router-dom';
import DetailBar from './DetailBar';

export default function HospitalDetailDrawer({ hospital, onClose, onOpenProfile }) {
  const navigate = useNavigate();

  if (!hospital) return null;

  return (
    <div className="w-full md:w-[400px] h-full flex flex-col bg-slate-900/80 border border-slate-800 rounded-xl shadow-sm overflow-hidden">
      <div className="p-5 flex flex-col h-full overflow-y-auto custom-scrollbar flex-1">
        
        {/* Drawer Header */}
        <div className="border-b border-slate-800/80 pb-4 mb-4 flex justify-between items-start shrink-0">
          <div>
            <h2 className="text-xl font-bold text-slate-100 uppercase tracking-tight pr-4 leading-tight">{hospital.name}</h2>
            <div className="flex flex-wrap gap-2 text-[10px] text-slate-400 font-mono mt-2 uppercase tracking-wider">
              <span>{hospital.area}</span>
              <span className="text-slate-600">•</span>
              <span>{hospital.hospital_type}</span>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="bg-slate-800/50 hover:bg-slate-700 text-slate-400 hover:text-white p-1.5 rounded-lg border border-slate-700/50 transition-colors shadow-sm focus:outline-none focus:ring-2 focus:ring-slate-500"
            aria-label="Close panel"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
          </button>
        </div>

        {/* Status Row */}
        <div className="mb-5 flex justify-between items-center bg-slate-950 p-3 rounded-lg border border-slate-800/80 shrink-0">
          <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Operational Status</span>
          <span className={`text-xs font-bold uppercase tracking-widest ${
            hospital.status !== 'Active' ? 'text-slate-500' :
            hospital.capacity_level === 'CRITICAL' ? 'text-red-500' :
            hospital.capacity_level === 'HIGH' ? 'text-orange-500' :
            'text-emerald-400'
          }`}>
            {hospital.status !== 'Active' ? 'OFFLINE' :
             hospital.capacity_level === 'CRITICAL' ? 'RESTRICTED' :
             hospital.capacity_level === 'HIGH' ? 'STANDBY' : 'OPERATIONAL'}
          </span>
        </div>

        <div className="flex flex-col gap-5 flex-1">
          {/* Bed Capacity */}
          <div>
            <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3 border-b border-slate-800/50 pb-1">Bed Capacity</h3>
            <div className="grid grid-cols-3 gap-2 mb-2">
              <div className="bg-slate-950 p-2 rounded border border-slate-800/50 flex flex-col">
                <span className="text-[9px] text-slate-500 uppercase tracking-widest mb-1">Total</span>
                <span className="text-sm font-mono text-slate-200">{hospital.total_beds ?? 'N/A'}</span>
              </div>
              <div className="bg-slate-950 p-2 rounded border border-slate-800/50 flex flex-col">
                <span className="text-[9px] text-slate-500 uppercase tracking-widest mb-1">Occupied</span>
                <span className="text-sm font-mono text-slate-300">{hospital.occupied_beds ?? 'N/A'}</span>
              </div>
              <div className="bg-slate-950 p-2 rounded border border-slate-800/50 flex flex-col">
                <span className="text-[9px] text-slate-500 uppercase tracking-widest mb-1">Available</span>
                <span className={`text-sm font-mono font-bold ${hospital.available_beds < 20 ? 'text-red-400' : 'text-emerald-400'}`}>{hospital.available_beds ?? 'N/A'}</span>
              </div>
            </div>
          </div>

          {/* Emergency Operations */}
          <div>
            <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3 border-b border-slate-800/50 pb-1">Emergency Operations</h3>
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-slate-950 p-2.5 rounded border border-slate-800/50 flex justify-between items-center">
                <span className="text-[9px] text-slate-500 uppercase tracking-widest">ER Occupancy</span>
                <span className="text-xs font-mono text-slate-300">{hospital.emergency_occupancy ?? 'N/A'}/{hospital.emergency_capacity ?? 'N/A'}</span>
              </div>
              <div className="bg-slate-950 p-2.5 rounded border border-slate-800/50 flex justify-between items-center">
                <span className="text-[9px] text-slate-500 uppercase tracking-widest">Wait Time</span>
                <span className="text-xs font-mono text-orange-400">{hospital.average_wait_minutes != null ? `${hospital.average_wait_minutes}m` : 'N/A'}</span>
              </div>
            </div>
          </div>

          {/* Compact Agent Card */}
          {hospital.agent ? (
            <div className="mt-auto bg-blue-950/20 border border-blue-900/40 rounded-xl overflow-hidden shadow-sm shrink-0">
               <div className="p-3 bg-blue-900/20 border-b border-blue-900/40 flex justify-between items-center">
                 <span className="text-[10px] text-blue-400 uppercase tracking-widest font-bold flex items-center gap-2">
                   <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse shadow-[0_0_6px_rgba(59,130,246,0.6)]"></span>
                   HOSPITAL AGENT
                 </span>
                 <span className="text-[9px] bg-blue-500/20 text-blue-300 px-1.5 py-0.5 rounded border border-blue-500/30 font-mono tracking-widest">ASSESSMENT</span>
               </div>
               <div className="p-4 flex flex-col gap-3">
                 <div className="flex justify-between items-center">
                   <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Current Condition</span>
                   <span className={`text-xs font-bold uppercase ${hospital.agent.current_condition === 'CRITICAL' ? 'text-red-400' : 'text-orange-400'}`}>{hospital.agent.current_condition}</span>
                 </div>
                 <div className="flex justify-between items-center">
                   <span className="text-[9px] text-slate-500 uppercase tracking-widest font-bold">Decision</span>
                   <span className="text-xs font-mono font-bold text-blue-400 truncate ml-4 text-right">{hospital.agent.decision_type}</span>
                 </div>
                 <button 
                   onClick={() => navigate('/hospital')} 
                   className="mt-2 w-full bg-blue-600/20 hover:bg-blue-600/40 text-blue-300 text-[10px] font-bold uppercase tracking-widest py-2 rounded border border-blue-500/30 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500"
                 >
                   View on Overview
                 </button>
               </div>
            </div>
          ) : (
             <div className="mt-auto bg-slate-950 p-4 rounded-xl border border-slate-800/80 flex flex-col items-center justify-center gap-1.5 shrink-0">
                <span className="text-[10px] text-slate-600 font-mono tracking-widest uppercase">Agent Assessment Unavailable</span>
             </div>
          )}

          {/* View Full Profile Action */}
          <button 
            onClick={onOpenProfile}
            className="shrink-0 mt-4 w-full bg-slate-800 hover:bg-slate-700 text-slate-300 text-[10px] font-bold uppercase tracking-widest py-3.5 rounded-lg border border-slate-700 transition-colors shadow-sm focus:outline-none focus:ring-2 focus:ring-slate-500"
          >
            View Full Profile
          </button>
        </div>
      </div>
    </div>
  );
}