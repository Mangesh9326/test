import React, { useState } from "react";
import { Outlet, NavLink } from "react-router-dom";
import { mainSidebar } from "../navigation/navigationConfig";
import {
  LayoutDashboard,
  Rocket,
  Box,
  TrafficCone,
  Hospital,
  Shield,
  Flame,
  Zap,
  Users,
  FileText,
  BrainCircuit,
  Settings,
  Search,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";

const icons = {
  dashboard: LayoutDashboard,
  launchpad: Rocket,
  "digital-twin": Box,
  traffic: TrafficCone,
  hospital: Hospital,
  police: Shield,
  fire: Flame,
  utility: Zap,
  citizen: Users,
  reports: FileText,
  analytics: BrainCircuit,
  settings: Settings,
};
// --- UI Icons ---
const IconSearch = () => (
  <svg
    className="w-4 h-4 text-slate-400"
    fill="none"
    viewBox="0 0 24 24"
    stroke="currentColor"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
    />
  </svg>
);
const IconChevronLeft = () => (
  <svg
    className="w-5 h-5"
    fill="none"
    viewBox="0 0 24 24"
    stroke="currentColor"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M15 19l-7-7 7-7"
    />
  </svg>
);
const IconChevronRight = () => (
  <svg
    className="w-5 h-5"
    fill="none"
    viewBox="0 0 24 24"
    stroke="currentColor"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M9 5l7 7-7 7"
    />
  </svg>
);

export default function MainLayout() {
  const [isCollapsed, setIsCollapsed] = useState(false);

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100 overflow-hidden relative font-sans">
      {/* --- UI/UX Enhancement: Subtle High-Tech OS Background --- */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:40px_40px] opacity-20 pointer-events-none z-0"></div>
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(59,130,246,0.15),rgba(255,255,255,0))] pointer-events-none z-0"></div>

      {/* GLOBAL HEADER */}
      <header className="h-16 bg-slate-900/80 backdrop-blur-md border-b border-slate-700/50 flex items-center justify-between px-6 z-20 shrink-0 shadow-lg">
        <div className="flex items-center gap-6">
          <span className="font-bold text-2xl bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-teal-300 tracking-tight">
            Smart CityOS
          </span>
          <div className="relative hidden md:block">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <IconSearch />
            </div>
            <input
              type="text"
              placeholder="Global Search..."
              className="bg-slate-950/50 border border-slate-700 rounded-lg pl-10 pr-4 py-1.5 text-sm text-slate-200 outline-none focus:border-blue-500 transition-colors w-64 shadow-inner"
            />
          </div>
        </div>
        <div className="flex items-center gap-5 text-sm font-medium">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.8)]"></span>
            <span className="text-emerald-400 tracking-wide uppercase text-xs">
              System Nominal
            </span>
          </div>
          <div className="w-px h-6 bg-slate-700"></div>
          <span className="text-slate-300 font-mono">72°F Clear</span>
          <button className="bg-blue-600/20 hover:bg-blue-600/40 text-blue-400 px-4 py-1.5 rounded-lg border border-blue-500/30 transition-colors shadow-sm font-bold tracking-wide">
            AI Assistant
          </button>
          <div className="w-9 h-9 rounded-full bg-indigo-500/20 border border-indigo-500/50 flex items-center justify-center text-indigo-300 font-bold shadow-md cursor-pointer hover:bg-indigo-500/40 transition-colors">
            A
          </div>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden relative z-10">
        {/* MAIN SIDEBAR (Level 1) - Collapsible */}
        <aside
          className={`flex-shrink-0 bg-slate-900/60 backdrop-blur-md border-r border-slate-700/50 flex flex-col transition-[width] duration-300 ease-in-out shadow-2xl z-20 ${
            isCollapsed ? "w-20" : "w-60"
          }`}
        >
          {/* Toggle Button */}
          <div
            className={`p-3 border-b border-slate-700/50 flex items-center ${isCollapsed ? "justify-center" : "justify-end"}`}
          >
            <button
              onClick={() => setIsCollapsed(!isCollapsed)}
              className="p-1.5 rounded-lg bg-slate-800/50 hover:bg-slate-700 text-slate-400 hover:text-white border border-slate-700/50 transition-colors focus:outline-none shadow-sm"
              title={isCollapsed ? "Expand Menu" : "Collapse Menu"}
            >
              {isCollapsed ? <IconChevronRight /> : <IconChevronLeft />}
            </button>
          </div>

          <nav className="p-3 flex flex-col gap-2 overflow-y-auto custom-scrollbar flex-1">
         

            {mainSidebar.map((item) => (
              <NavLink
                key={item.title}
                to={item.path}
                end={item.path === "/"}
                title={isCollapsed ? item.title : undefined}
                className={({ isActive }) =>
                  `group relative flex items-center rounded-lg text-sm font-medium
      transition-all duration-200 overflow-hidden whitespace-nowrap
      ${
        isCollapsed
          ? "justify-center w-full h-12 px-2"
          : "w-full px-4 py-2.5 gap-3"
      }
      ${
        isActive
          ? "bg-blue-600/20 text-blue-400 border border-blue-500/30 shadow-inner"
          : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/80 border border-transparent"
      }`
                }
              >
                {({ isActive }) => (
                  <>
           {isCollapsed ? (
  <span
    className={`flex items-center justify-center shrink-0 ${
      isActive
        ? "text-blue-400"
        : "text-slate-400 group-hover:text-blue-300"
    }`}
  >
    {(() => {
      const Icon = icons[item.icon];

      return Icon ? (
        <Icon
          size={20}
          strokeWidth={1.8}
          aria-hidden="true"
        />
      ) : null;
    })()}
  </span>
) : (
  <span className="truncate w-full">
    {item.title}
  </span>
)}
                  </>
                )}
              </NavLink>
            ))}
          </nav>
        </aside>

        {/* DYNAMIC CENTER CONTENT */}
        <main className="flex-1 flex flex-col overflow-hidden relative transition-all duration-300 ease-in-out bg-slate-950/40 backdrop-blur-[2px]">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
