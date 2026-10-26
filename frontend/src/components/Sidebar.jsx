import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';

// --- UI Icons ---
const IconDashboard = () => <svg className="w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zm10 0a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zm10 0a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" /></svg>;
const IconTraffic = () => <svg className="w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" /></svg>;
const IconHospital = () => <svg className="w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" /></svg>;
const IconPolice = () => <svg className="w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" /></svg>;
const IconFire = () => <svg className="w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17.657 18.657A8 8 0 116.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.657 7.343a7.975 7.975 0 010 11.314z" /></svg>;
const IconUtility = () => <svg className="w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>;
const IconChevronLeft = () => <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7" /></svg>;
const IconChevronRight = () => <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" /></svg>;

const Sidebar = () => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const navItems = [
    { name: 'Command Center', path: '/', icon: <IconDashboard /> },
    { name: 'Traffic Control', path: '/traffic', icon: <IconTraffic /> },
    { name: 'Hospital Logistics', path: '/hospital', icon: <IconHospital /> },
    { name: 'Police Dispatch', path: '/police', icon: <IconPolice /> },
    { name: 'Fire & Rescue', path: '/fire', icon: <IconFire /> },
    { name: 'Utility Grid', path: '/utility', icon: <IconUtility /> },
  ];

  return (
    <aside 
      className={`h-screen glass-panel rounded-none border-y-0 border-l-0 border-city-700/50 flex flex-col transition-[width] duration-300 ease-in-out ${
        isCollapsed ? 'w-20' : 'w-64'
      }`}
    >
      {/* Header & Toggle */}
      <div className={`p-5 border-b border-city-700/50 flex items-center h-20 shrink-0 ${isCollapsed ? 'justify-center' : 'justify-between'}`}>
        {!isCollapsed && (
          <div className="overflow-hidden whitespace-nowrap">
            <h2 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-teal-300">
              Smart CityOS
            </h2>
            <p className="text-[10px] text-gray-400 uppercase tracking-widest mt-0.5 font-mono">v2.0.26</p>
          </div>
        )}
        
        <button 
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="p-1.5 rounded-lg bg-city-800/50 hover:bg-city-700 text-gray-400 hover:text-white border border-city-700/50 transition-colors focus:outline-none"
          title={isCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
        >
          {isCollapsed ? <IconChevronRight /> : <IconChevronLeft />}
        </button>
      </div>
      
      {/* Navigation */}
      <nav className="flex-1 p-3 flex flex-col gap-2 overflow-y-auto custom-scrollbar overflow-x-hidden">
        <div className={`text-[10px] font-bold text-gray-500 uppercase tracking-wider mb-2 mt-2 transition-all duration-300 ${isCollapsed ? 'text-center' : 'px-3'}`}>
          {isCollapsed ? 'MOD' : 'Modules'}
        </div>
        
        {navItems.map((item) => (
          <NavLink
            key={item.name}
            to={item.path}
            title={isCollapsed ? item.name : ''} // Tooltip for collapsed mode
            className={({ isActive }) =>
              `flex items-center rounded-lg text-sm font-medium transition-all duration-200 overflow-hidden whitespace-nowrap ${
                isCollapsed ? 'justify-center p-3' : 'px-4 py-3 gap-3'
              } ${
                isActive
                  ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30 shadow-[0_0_10px_rgba(59,130,246,0.1)]'
                  : 'text-gray-400 hover:bg-city-800 hover:text-gray-200 border border-transparent'
              }`
            }
          >
            {item.icon}
            {!isCollapsed && <span>{item.name}</span>}
          </NavLink>
        ))}
      </nav>

      {/* User Profile Section */}
      <div className={`p-4 border-t border-city-700/50 flex items-center shrink-0 ${isCollapsed ? 'justify-center' : 'gap-3 px-4'}`}>
        <div className="w-9 h-9 shrink-0 rounded-full bg-indigo-500/20 border border-indigo-500/50 flex items-center justify-center text-indigo-300 font-bold shadow-lg">
          A
        </div>
        {!isCollapsed && (
          <div className="overflow-hidden whitespace-nowrap">
            <div className="text-sm font-bold text-gray-200 truncate">Admin User</div>
            <div className="text-[10px] text-gray-500 uppercase tracking-wide truncate">System Admin</div>
          </div>
        )}
      </div>
    </aside>
  );
};

export default Sidebar;