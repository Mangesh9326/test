import React, { useCallback, useEffect, useRef, useState } from "react";
import { Outlet, NavLink, useLocation } from "react-router-dom";
import { useCitizenStore } from "../../store/useCitizenStore";

// ---------------------------------------------------------------------------
// Icons (inline SVG, decorative: the visible label carries the meaning)
// ---------------------------------------------------------------------------
const Svg = ({ d, className = "w-6 h-6" }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d={d} />
  </svg>
);

const Icons = {
  Map: () => <Svg d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7l6-3 5.447 2.724A1 1 0 0121 7.618v10.764a1 1 0 01-1.447.894L15 17l-6 3z" />,
  Traffic: () => <Svg d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />,
  Alerts: () => <Svg d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />,
  Services: () => <Svg d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />,
  AI: () => <Svg d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />,
  Pin: () => <Svg className="w-5 h-5" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0zM15 11a3 3 0 11-6 0 3 3 0 016 0z" />,
  User: () => <Svg className="w-5 h-5" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />,
  Menu: () => <Svg d="M4 6h16M4 12h16M4 18h16" />,
  Close: () => <Svg d="M6 18L18 6M6 6l12 12" />,
  Chevron: () => <Svg className="w-4 h-4" d="M9 5l7 7-7 7" />,
};

// Single source of truth for navigation.
// `label` = short text for the bottom bar / rail, `title` = full name for the drawer.
const NAV_ITEMS = [
  { path: "/citizen/around-me", label: "Around", title: "Around Me", desc: "Nearby places & live map", icon: Icons.Map },
  { path: "/citizen/traffic", label: "Traffic", title: "Traffic", desc: "Roads, delays & routes", icon: Icons.Traffic },
  { path: "/citizen/alerts", label: "Alerts", title: "Alerts", desc: "Warnings & notices", icon: Icons.Alerts, hasBadge: true },
  { path: "/citizen/services", label: "Services", title: "City Services", desc: "Report issues & get help", icon: Icons.Services },
  { path: "/citizen/assistant", label: "AI", title: "AI City Assistant", desc: "Ask anything about the city", icon: Icons.AI },
];

const focusRing = "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400/80";
const badgeText = (n) => (n > 9 ? "9+" : n);

// Tracks browser connectivity so we can tell people when live data is paused
function useOnline() {
  const [online, setOnline] = useState(typeof navigator === "undefined" ? true : navigator.onLine);
  useEffect(() => {
    const up = () => setOnline(true);
    const down = () => setOnline(false);
    window.addEventListener("online", up);
    window.addEventListener("offline", down);
    return () => {
      window.removeEventListener("online", up);
      window.removeEventListener("offline", down);
    };
  }, []);
  return online;
}

export default function CitizenLayout() {
  const locationState = useCitizenStore((state) => state.location);
  const alertCount = useCitizenStore((state) => state.alerts?.length || 0);

  const { pathname } = useLocation();
  const mainRef = useRef(null);
  const menuBtnRef = useRef(null);
  const closeBtnRef = useRef(null);
  const drawerRef = useRef(null);
  const online = useOnline();
  const [menuOpen, setMenuOpen] = useState(false);

  const closeMenu = useCallback(() => {
    setMenuOpen(false);
    menuBtnRef.current?.focus();
  }, []);

  // Each page starts at the top when you switch tabs
  useEffect(() => {
    mainRef.current?.scrollTo({ top: 0 });
  }, [pathname]);

  // Close the drawer whenever the route changes
  useEffect(() => {
    setMenuOpen(false);
  }, [pathname]);

  // Drawer behaviour: Escape closes, Tab stays inside, focus moves in on open
  useEffect(() => {
    if (!menuOpen) return;
    closeBtnRef.current?.focus();

    const onKey = (e) => {
      if (e.key === "Escape") {
        closeMenu();
        return;
      }
      if (e.key !== "Tab" || !drawerRef.current) return;
      const focusables = drawerRef.current.querySelectorAll("a[href], button:not([disabled])");
      if (!focusables.length) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [menuOpen, closeMenu]);

  // If the window grows past the mobile breakpoint, make sure the drawer is closed
  useEffect(() => {
    const mq = window.matchMedia("(min-width: 768px)");
    const onChange = (e) => e.matches && setMenuOpen(false);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  const areaName = locationState?.name || "Mumbai";

  return (
    <div className="flex h-[100dvh] justify-center bg-slate-950 font-sans">
      <style>{`
        @keyframes cl-fade{from{opacity:0}to{opacity:1}}
        .cl-page{animation:cl-fade .18s ease-out}
        @media (prefers-reduced-motion:reduce){.cl-page{animation:none}}
      `}</style>

      <a
        href="#citizen-main"
        className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-[60] focus:rounded-lg focus:bg-blue-600 focus:px-3 focus:py-2 focus:text-sm focus:font-semibold focus:text-white"
      >
        Skip to content
      </a>

      {/* Shell: phone-sized column on small screens, full width with a side rail from md up.
          To keep the phone-only look on every screen, remove "md:max-w-none" and the md: classes. */}
      <div className="relative flex h-full w-full max-w-[430px] flex-col overflow-hidden bg-slate-900 shadow-2xl sm:border-x sm:border-slate-800 md:max-w-none md:flex-row md:border-x-0">
        {/* Content column: header, status banner, page.
            min-h-0 is what lets <main> shrink and scroll inside a flex column. */}
        <div className="flex min-h-0 min-w-0 flex-1 flex-col md:order-2">
          <header className="z-30 flex shrink-0 items-center justify-between gap-2 border-b border-slate-800 bg-slate-900/95 px-2 pb-2.5 pt-[max(0.625rem,env(safe-area-inset-top))] backdrop-blur-md md:px-6">
            <div className="flex min-w-0 items-center gap-1">
              {/* Mobile menu button */}
              <button
                ref={menuBtnRef}
                type="button"
                onClick={() => setMenuOpen(true)}
                aria-label="Open navigation menu"
                aria-expanded={menuOpen}
                aria-controls="citizen-drawer"
                className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-full text-slate-300 transition-colors hover:bg-slate-800 hover:text-white motion-safe:active:scale-95 md:hidden ${focusRing}`}
              >
                <Icons.Menu />
              </button>

              <div className="flex min-w-0 items-center gap-2.5 md:pl-0">
                <span className="hidden h-9 w-9 shrink-0 items-center justify-center rounded-full bg-blue-500/10 text-blue-400 min-[360px]:flex">
                  <Icons.Pin />
                </span>
                <div className="flex min-w-0 flex-col leading-tight">
                  <span className="text-[11px] text-slate-400">Current area</span>
                  <span className="max-w-[40vw] truncate text-sm font-semibold text-slate-100 sm:max-w-xs">
                    {areaName}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex shrink-0 items-center gap-1 text-slate-300">
              {/* The bell already lives in the side rail on larger screens */}
              <NavLink
                to="/citizen/alerts"
                aria-label={alertCount > 0 ? `Alerts, ${alertCount} active` : "Alerts"}
                className={`relative flex h-11 w-11 items-center justify-center rounded-full hover:bg-slate-800 hover:text-white md:hidden ${focusRing}`}
              >
                <Icons.Alerts />
                {alertCount > 0 && (
                  <span className="absolute right-1 top-1 flex h-4 min-w-[16px] items-center justify-center rounded-full border border-slate-900 bg-red-500 px-1 text-[10px] font-bold leading-none text-white">
                    {badgeText(alertCount)}
                  </span>
                )}
              </NavLink>
              <button
                type="button"
                aria-label="Profile"
                className={`flex h-11 w-11 items-center justify-center rounded-full hover:bg-slate-800 hover:text-white ${focusRing}`}
              >
                <span className="flex h-8 w-8 items-center justify-center rounded-full border border-slate-600 bg-slate-700">
                  <Icons.User />
                </span>
              </button>
            </div>
          </header>

          {!online && (
            <div role="status" className="shrink-0 border-b border-amber-500/30 bg-amber-500/10 px-4 py-1.5 text-center text-xs font-medium text-amber-300">
              You're offline. Live updates are paused until you reconnect.
            </div>
          )}

          {/* flex-1 + min-h-0 + overflow-y-auto = the actual scroll container */}
          <main
            id="citizen-main"
            ref={mainRef}
            tabIndex={-1}
            className="custom-scrollbar relative min-h-0 flex-1 overflow-y-auto overflow-x-hidden overscroll-contain scroll-smooth focus:outline-none"
          >
            {/* Opacity-only transition on purpose: a transform here would trap fixed drawers inside pages.
                min-h-full (not h-full) so pages taller than the screen can grow and scroll. */}
            <div key={pathname} className="cl-page min-h-full">
              <Outlet />
            </div>
          </main>
        </div>

        {/* Navigation: bottom bar on phones, left rail from md (icons only), labelled rail from lg */}
        <nav
          aria-label="Primary"
          className="z-30 shrink-0 border-t border-slate-800 bg-slate-900/95 pb-[env(safe-area-inset-bottom)] backdrop-blur-md md:order-1 md:w-20 md:border-r md:border-t-0 md:pb-0 lg:w-56"
        >
          <div className="hidden items-center gap-2 px-4 pb-2 pt-5 md:flex lg:px-5">
            <span className="text-blue-400"><Icons.Pin /></span>
            <span className="hidden text-sm font-semibold text-slate-100 lg:inline">Citizen</span>
          </div>

          <ul className="grid grid-cols-5 gap-1 px-2 py-1.5 md:flex md:flex-col md:gap-1 md:p-3">
            {NAV_ITEMS.map((item) => {
              const badge = item.hasBadge ? alertCount : 0;
              return (
                <li key={item.path} className="min-w-0">
                  <NavLink
                    to={item.path}
                    title={item.title}
                    className={({ isActive }) =>
                      `relative flex h-14 flex-col items-center justify-center gap-0.5 rounded-xl px-1 transition-colors motion-safe:active:scale-95 lg:h-11 lg:flex-row lg:justify-start lg:gap-3 lg:px-3 ${focusRing} ${
                        isActive ? "bg-blue-500/10 text-blue-400" : "text-slate-400 hover:bg-slate-800/60 hover:text-slate-200"
                      }`
                    }
                  >
                    {({ isActive }) => (
                      <>
                        {isActive && (
                          <span aria-hidden="true" className="absolute inset-x-5 top-0 h-0.5 rounded-full bg-blue-400 md:inset-x-auto md:inset-y-3 md:-left-3 md:h-auto md:w-0.5" />
                        )}
                        <span className="relative">
                          <item.icon />
                          {badge > 0 && (
                            <span className="absolute -right-2 -top-1.5 flex h-4 min-w-[16px] items-center justify-center rounded-full border border-slate-900 bg-red-500 px-1 text-[10px] font-bold leading-none text-white">
                              <span aria-hidden="true">{badgeText(badge)}</span>
                              <span className="sr-only">{badge} active</span>
                            </span>
                          )}
                        </span>
                        <span className="truncate text-[11px] font-semibold md:text-[10px] lg:text-sm lg:font-medium">{item.label}</span>
                      </>
                    )}
                  </NavLink>
                </li>
              );
            })}
          </ul>
        </nav>

        {/* ---------------- Mobile drawer ----------------
            Lives inside the shell (absolute, not fixed) so it stays within the phone column.
            Kept outside the header because backdrop-blur would trap it there. */}
        <div
          className={`absolute inset-0 z-50 md:hidden ${menuOpen ? "visible" : "invisible"} transition-[visibility] duration-300`}
          aria-hidden={!menuOpen}
        >
          {/* Backdrop */}
          <button
            type="button"
            tabIndex={-1}
            aria-label="Close navigation menu"
            onClick={closeMenu}
            className={`absolute inset-0 bg-slate-950/70 backdrop-blur-sm transition-opacity duration-300 ${menuOpen ? "opacity-100" : "opacity-0"}`}
          />

          {/* Panel */}
          <div
            id="citizen-drawer"
            ref={drawerRef}
            role="dialog"
            aria-modal="true"
            aria-label="Navigation menu"
            className={`absolute inset-y-0 left-0 flex w-[82%] max-w-[320px] flex-col border-r border-slate-800 bg-slate-900 shadow-2xl transition-transform duration-300 ease-out motion-reduce:transition-none ${
              menuOpen ? "translate-x-0" : "-translate-x-full"
            }`}
          >
            <div className="flex shrink-0 items-center justify-between border-b border-slate-800 px-4 pb-3 pt-[max(0.75rem,env(safe-area-inset-top))]">
              <div className="flex items-center gap-2.5">
                <span className="flex h-9 w-9 items-center justify-center rounded-full bg-blue-500/10 text-blue-400">
                  <Icons.Pin />
                </span>
                <div className="flex flex-col leading-tight">
                  <span className="text-sm font-semibold text-slate-100">Citizen</span>
                  <span className="max-w-[160px] truncate text-[11px] text-slate-400">{areaName}</span>
                </div>
              </div>
              <button
                ref={closeBtnRef}
                type="button"
                onClick={closeMenu}
                aria-label="Close navigation menu"
                className={`flex h-11 w-11 items-center justify-center rounded-full text-slate-300 hover:bg-slate-800 hover:text-white motion-safe:active:scale-95 ${focusRing}`}
              >
                <Icons.Close />
              </button>
            </div>

            <nav aria-label="Menu" className="custom-scrollbar min-h-0 flex-1 overflow-y-auto overscroll-contain p-3">
              <ul className="flex flex-col gap-1">
                {NAV_ITEMS.map((item) => {
                  const badge = item.hasBadge ? alertCount : 0;
                  return (
                    <li key={item.path}>
                      <NavLink
                        to={item.path}
                        onClick={() => setMenuOpen(false)}
                        className={({ isActive }) =>
                          `flex items-center gap-3 rounded-xl px-3 py-3 transition-colors motion-safe:active:scale-[0.98] ${focusRing} ${
                            isActive
                              ? "bg-blue-500/10 text-blue-400 ring-1 ring-inset ring-blue-500/20"
                              : "text-slate-300 hover:bg-slate-800/70 hover:text-white"
                          }`
                        }
                      >
                        <span className="relative flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-slate-800/80">
                          <item.icon />
                        </span>
                        <span className="flex min-w-0 flex-1 flex-col leading-tight">
                          <span className="truncate text-sm font-semibold">{item.title}</span>
                          <span className="truncate text-xs text-slate-400">{item.desc}</span>
                        </span>
                        {badge > 0 ? (
                          <span className="flex h-5 min-w-[20px] items-center justify-center rounded-full bg-red-500 px-1.5 text-[11px] font-bold leading-none text-white">
                            <span aria-hidden="true">{badgeText(badge)}</span>
                            <span className="sr-only">{badge} active</span>
                          </span>
                        ) : (
                          <span className="text-slate-600"><Icons.Chevron /></span>
                        )}
                      </NavLink>
                    </li>
                  );
                })}
              </ul>
            </nav>

            <div className="shrink-0 border-t border-slate-800 px-4 pb-[max(0.75rem,env(safe-area-inset-bottom))] pt-3">
              <p className="flex items-center gap-2 text-xs text-slate-400">
                <span
                  aria-hidden="true"
                  className={`h-2 w-2 rounded-full ${online ? "bg-emerald-400" : "bg-amber-400"}`}
                />
                {online ? "Live updates on" : "Offline: live updates paused"}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}