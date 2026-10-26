import React from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';

export default function MainLayout() {
  return (
    <div className="flex h-screen w-full bg-slate-950 text-slate-100 overflow-hidden relative font-sans">
      
      {/* --- UI/UX Enhancement: Subtle High-Tech OS Background --- */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:40px_40px] opacity-20 pointer-events-none z-0"></div>
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(59,130,246,0.15),rgba(255,255,255,0))] pointer-events-none z-0"></div>

      {/* Global Sidebar Navigation */}
      {/* shrink-0 prevents it from squishing, allowing Sidebar's internal width transition to push the layout smoothly */}
      <div className="z-20 h-full shrink-0 shadow-2xl flex">
        <Sidebar />
      </div>
      
      {/* Dynamic Page Content */}
      {/* Changed to overflow-hidden so nested department sidebars stay fixed on screen! */}
      <main className="flex-1 h-screen overflow-hidden relative z-10 bg-slate-950/40 backdrop-blur-sm transition-all duration-300">
        <Outlet />
      </main>
      
    </div>
  );
}