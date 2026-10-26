import React from 'react';
import { Outlet, NavLink } from 'react-router-dom';

export default function DepartmentLayout({ title, sidebarConfig }) {
  return (
    <div className="flex h-full w-full">
      
      {/* SECONDARY DEPARTMENT SIDEBAR (Fixed Width - Does Not Collapse) */}
      <aside className="w-60 shrink-0 glass-panel rounded-none border-y-0 border-l-0 bg-slate-900/60 backdrop-blur-md flex flex-col z-10 shadow-2xl">
        
        {/* Department Header */}
        <div className="p-5 border-b border-slate-700/50 flex items-center gap-3 shrink-0">
          <div className="w-2.5 h-2.5 rounded-full bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.8)] animate-pulse"></div>
          <h3 className="font-bold text-slate-100 tracking-wide text-lg">{title} Ops</h3>
        </div>
        
        {/* Department Navigation */}
        <nav className="p-4 flex flex-col gap-2 overflow-y-auto custom-scrollbar flex-1">
          {sidebarConfig.map((item) => (
            <NavLink 
              key={item.title} 
              to={item.path}
              end={item.path === `/${title.toLowerCase()}`}
              className={({ isActive }) => 
                `px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                  isActive 
                    ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30 shadow-inner' 
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/80 border border-transparent'
                }`
              }
            >
              {item.title}
            </NavLink>
          ))}
        </nav>
      </aside>

      {/* DEPARTMENT CONTENT AREA (Scrolls independently) */}
      <section className="flex-1 h-full overflow-y-auto overflow-x-hidden p-6 custom-scrollbar relative z-0">
        <Outlet />
      </section>
      
    </div>
  );
}