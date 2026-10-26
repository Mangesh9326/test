import React, { useEffect } from 'react';
import DetailBar from './DetailBar';

export default function HospitalProfileModal({ hospital, onClose, onPrev, onNext, hasPrev, hasNext }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    document.body.style.overflow = 'hidden'; 
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = '';
    };
  }, [onClose]);

  if (!hospital) return null;

  let opStatus = "OPERATIONAL";
  let opColor = "text-emerald-400 bg-emerald-500/10 border-emerald-500/30";
  if (hospital.status !== 'Active') {
    opStatus = "OFFLINE";
    opColor = "text-slate-400 bg-slate-800 border-slate-700";
  } else if (hospital.capacity_level === 'CRITICAL') {
    opStatus = "RESTRICTED";
    opColor = "text-red-400 bg-red-500/10 border-red-500/30";
  } else if (hospital.capacity_level === 'HIGH') {
    opStatus = "STANDBY";
    opColor = "text-orange-400 bg-orange-500/10 border-orange-500/30";
  }

  let emerStatus = "ACCEPTING";
  let emerColor = "text-emerald-400 bg-emerald-500/10 border-emerald-500/30";
  if (!hospital.emergency_available) {
    emerStatus = "UNAVAILABLE";
    emerColor = "text-slate-500 bg-slate-800 border-slate-700";
  } else if (hospital.emergency_level === 'CRITICAL' || hospital.emergency_level === 'HIGH') {
    emerStatus = "RESTRICTED";
    emerColor = "text-orange-400 bg-orange-500/10 border-orange-500/30";
  }

  const hasResources = hospital.oxygen_status || hospital.ventilators_total || hospital.blood_supply_status || hospital.diagnostic_capacity || hospital.operating_rooms;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-0 sm:p-6 bg-slate-950/80 backdrop-blur-sm">
      <div 
        className="bg-slate-900 border border-slate-700 w-full h-full sm:h-auto sm:max-h-[90vh] sm:max-w-4xl sm:rounded-xl shadow-2xl flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-200"
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
      >
        <div className="bg-slate-950 border-b border-slate-800 p-5 shrink-0 flex justify-between items-start gap-4">
          <div>
            <h2 id="modal-title" className="text-2xl font-bold text-white uppercase tracking-tight">{hospital.name}</h2>
            <div className="flex items-center gap-2 text-xs text-slate-400 font-mono mt-1">
              <span>{hospital.hospital_id.split('-')[0].toUpperCase()}</span>
              <span className="text-slate-600">•</span>
              <span>{hospital.area}</span>
              <span className="text-slate-600">•</span>
              <span className="uppercase tracking-wider">{hospital.hospital_type}</span>
            </div>
            <div className="flex flex-wrap items-center gap-3 mt-4">
              <span className={`px-2.5 py-1 text-[10px] font-bold uppercase tracking-widest rounded border ${opColor} flex items-center gap-1.5`}>
                <span className={`w-1.5 h-1.5 rounded-full ${opStatus === 'OPERATIONAL' ? 'bg-emerald-500' : opStatus === 'OFFLINE' ? 'bg-slate-500' : 'bg-current'}`}></span>
                {opStatus}
              </span>
              <span className={`px-2.5 py-1 text-[10px] font-bold uppercase tracking-widest rounded border ${emerColor}`}>
                EMERGENCY: {emerStatus}
              </span>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 rounded-lg border border-slate-700 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-400"
            aria-label="Close profile"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar p-5 sm:p-6 bg-slate-900">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="space-y-8">
              <section>
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3 border-b border-slate-800 pb-2">Bed Capacity</h3>
                <div className="grid grid-cols-3 gap-3 mb-3">
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase tracking-widest block mb-1">Total</span>
                    <span className="text-lg font-mono text-slate-200">{hospital.total_beds ?? 'DATA UNAVAILABLE'}</span>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase tracking-widest block mb-1">Occupied</span>
                    <span className="text-lg font-mono text-slate-300">{hospital.occupied_beds ?? 'DATA UNAVAILABLE'}</span>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase tracking-widest block mb-1">Available</span>
                    <span className={`text-lg font-mono font-bold ${hospital.available_beds < 20 ? 'text-red-400' : 'text-emerald-400'}`}>{hospital.available_beds ?? 'DATA UNAVAILABLE'}</span>
                  </div>
                </div>
                <DetailBar label="Overall Bed Occupancy" occupied={hospital.occupied_beds} total={hospital.total_beds} colorClass={(hospital.occupied_beds / hospital.total_beds) > 0.9 ? 'bg-red-500' : 'bg-emerald-500'} />
              </section>

              <section>
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3 border-b border-slate-800 pb-2">ICU Capacity</h3>
                <div className="grid grid-cols-3 gap-3 mb-3">
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase tracking-widest block mb-1">Total</span>
                    <span className="text-lg font-mono text-slate-200">{hospital.icu_total ?? 'DATA UNAVAILABLE'}</span>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase tracking-widest block mb-1">Occupied</span>
                    <span className="text-lg font-mono text-slate-300">{hospital.icu_occupied ?? 'DATA UNAVAILABLE'}</span>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase tracking-widest block mb-1">Available</span>
                    <span className={`text-lg font-mono font-bold ${hospital.icu_available <= 5 ? 'text-red-400' : 'text-blue-400'}`}>{hospital.icu_available ?? 'DATA UNAVAILABLE'}</span>
                  </div>
                </div>
                <DetailBar label="ICU Occupancy" occupied={hospital.icu_occupied} total={hospital.icu_total} colorClass={(hospital.icu_occupied / hospital.icu_total) > 0.85 ? 'bg-red-500' : 'bg-blue-500'} />
              </section>
            </div>

            <div className="space-y-8">
              <section>
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3 border-b border-slate-800 pb-2">Emergency Operations</h3>
                <div className="grid grid-cols-2 gap-3 mb-3">
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase tracking-widest block mb-1">ER Load</span>
                    <span className="text-sm font-mono text-slate-300">{hospital.emergency_occupancy ?? 'DATA UNAVAILABLE'} / {hospital.emergency_capacity ?? 'N/A'}</span>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase tracking-widest block mb-1">Avg Wait</span>
                    <span className="text-sm font-mono text-orange-400">{hospital.average_wait_minutes != null ? `${hospital.average_wait_minutes} min` : 'DATA UNAVAILABLE'}</span>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 col-span-2 flex justify-between items-center">
                    <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Ambulance Queue</span>
                    <span className="text-lg font-mono font-bold text-red-400">{hospital.ambulance_queue ?? 'DATA UNAVAILABLE'}</span>
                  </div>
                </div>
                <DetailBar label="ER Occupancy" occupied={hospital.emergency_occupancy} total={hospital.emergency_capacity} colorClass={(hospital.emergency_occupancy / hospital.emergency_capacity) > 0.85 ? 'bg-red-500' : 'bg-orange-500'} />
              </section>

              <section>
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3 border-b border-slate-800 pb-2">Patient Flow</h3>
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase tracking-widest block mb-1">Current Inflow</span>
                    <span className="text-sm font-bold text-slate-300 uppercase">{hospital.patient_inflow || 'DATA UNAVAILABLE'}</span>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase tracking-widest block mb-1">Predicted</span>
                    <span className="text-sm font-bold text-blue-400 uppercase">{hospital.predicted_inflow || 'DATA UNAVAILABLE'}</span>
                  </div>
                </div>
              </section>

              <section>
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-3 border-b border-slate-800 pb-2">Resource Readiness</h3>
                {hasResources ? (
                  <div className="grid grid-cols-2 gap-3">
                    {hospital.oxygen_status && <div className="bg-slate-950 p-3 rounded-lg border border-slate-800"><span className="text-[10px] text-slate-500 uppercase tracking-widest block mb-1">Oxygen</span><span className="text-sm font-bold text-slate-300 uppercase">{hospital.oxygen_status}</span></div>}
                    {hospital.ventilators_total && <div className="bg-slate-950 p-3 rounded-lg border border-slate-800"><span className="text-[10px] text-slate-500 uppercase tracking-widest block mb-1">Ventilators</span><span className="text-sm font-mono text-slate-300">{hospital.ventilators_available} / {hospital.ventilators_total}</span></div>}
                  </div>
                ) : (
                  <div className="bg-slate-950/80 p-6 rounded-lg border border-slate-800 flex flex-col items-center justify-center text-center gap-2">
                    <span className="text-xs text-slate-500 font-mono tracking-widest uppercase">Data Unavailable</span>
                  </div>
                )}
              </section>
            </div>
          </div>
        </div>

        <div className="bg-slate-950 border-t border-slate-800 p-4 shrink-0 flex flex-col sm:flex-row justify-between items-center gap-4">
          <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">
            Last Updated: {hospital.last_observed_at ? new Date(hospital.last_observed_at).toLocaleTimeString() : 'DATA UNAVAILABLE'}
          </span>
          <div className="flex gap-2 w-full sm:w-auto">
            <button 
              onClick={onPrev} 
              disabled={!hasPrev}
              className="flex-1 sm:flex-none px-4 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-300 text-[10px] font-bold uppercase tracking-widest rounded border border-slate-700 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-400"
            >
              Previous
            </button>
            <button 
              onClick={onNext} 
              disabled={!hasNext}
              className="flex-1 sm:flex-none px-4 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-300 text-[10px] font-bold uppercase tracking-widest rounded border border-slate-700 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-400"
            >
              Next
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}