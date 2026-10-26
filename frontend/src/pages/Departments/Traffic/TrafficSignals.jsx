import React, { useState, useEffect, useCallback } from 'react';
import SignalMonitoringTable from './components/SignalMonitoringTable';

const LEVEL_TEXT = {
  CRITICAL: 'text-red-500',
  SEVERE: 'text-red-400',
  HIGH: 'text-orange-400',
  MODERATE: 'text-yellow-400',
  LOW: 'text-emerald-400',
};

export default function TrafficSignals() {
  const [signals, setSignals] = useState([]);
  const [summary, setSummary] = useState(null);
  const [areaStats, setAreaStats] = useState([]);
  const [activityFeed, setActivityFeed] = useState([]);
  const [isTableExpanded, setIsTableExpanded] = useState(false);

  const [selectedSignal, setSelectedSignal] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [apiError, setApiError] = useState(false);

  // Pagination & Filters
  const [page, setPage] = useState(1);
  const [limit, setLimit] = useState(50);
  const [totalSignals, setTotalSignals] = useState(0);
  const [filters, setFilters] = useState({
    search: '', area: 'All', traffic: 'All', density: 'All', status: 'All', decision: 'All'
  });
  const [filtersOpen, setFiltersOpen] = useState(false);

  const fetchTrafficData = useCallback(async () => {
    try {
      setApiError(false);
      const queryParams = new URLSearchParams({
        page, limit,
        area: filters.area, traffic: filters.traffic, status: filters.status,
        decision: filters.decision, search: filters.search
      }).toString();

      const [signalsRes, summaryRes, areasRes, activityRes] = await Promise.all([
        fetch(`/api/traffic/signals?${queryParams}`),
        fetch('/api/traffic/signals/summary'),
        fetch('/api/traffic/signals/areas'),
        fetch('/api/traffic/signals/activity')
      ]);

      if (!signalsRes.ok) throw new Error('API Error');

      const signalsData = await signalsRes.json();
      setSignals(signalsData.signals);
      setTotalSignals(signalsData.pagination.total);

      if (summaryRes.ok) setSummary(await summaryRes.json());
      if (areasRes.ok) setAreaStats(await areasRes.json());
      if (activityRes.ok) setActivityFeed(await activityRes.json());

      setLastUpdated(new Date());

      // Auto-select first signal if none selected
      if (!selectedSignal && signalsData.signals.length > 0) {
        setSelectedSignal(signalsData.signals[0]);
      }
    } catch (err) {
      console.error("Failed to load signal network:", err);
      setApiError(true);
    }
  }, [page, limit, filters, selectedSignal]);

  // Polling Mechanism (30s)
  useEffect(() => {
    fetchTrafficData();
    const interval = setInterval(fetchTrafficData, 30000);
    return () => clearInterval(interval);
  }, [fetchTrafficData]);

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
    setPage(1); // Reset to page 1 on filter change
  };

  const clearFilters = () => {
    setFilters({ search: '', area: 'All', traffic: 'All', density: 'All', status: 'All', decision: 'All' });
    setPage(1);
  };

  const activeFilterCount = Object.entries(filters).filter(([k, v]) => v && v !== 'All').length;

  // Signal Timing Bar Component
  const renderTimingBar = (green, red) => {
    if (green == null || red == null) return null;
    const total = green + red || 1;
    return (
      <div className="flex h-2.5 w-full rounded-full overflow-hidden bg-slate-800 border border-slate-700">
        <div className="bg-emerald-500" style={{ width: `${(green / total) * 100}%` }} />
        <div className="bg-red-500" style={{ width: `${(red / total) * 100}%` }} />
      </div>
    );
  };

  // Dims zero values so non-zero counts stand out in dense tables
  const renderCount = (value, colorClass) => {
    const v = value || 0;
    return v === 0 ? <span className="text-slate-700">0</span> : <span className={`font-bold ${colorClass}`}>{v}</span>;
  };

  // Derived attention signals for top cards
  const attentionSignals = signals.filter(s => ['CRITICAL', 'SEVERE', 'HIGH'].includes(s.traffic_level)).slice(0, 6);

  // Replaces "Active Deployments" with Active Conditions
  const activeConditions = signals.filter(s => ['CRITICAL', 'SEVERE'].includes(s.traffic_level)).slice(0, 4);

  const kpis = [
    { label: 'Signals monitored', val: summary?.totalSignals ?? 0, color: 'text-slate-100' },
    { label: 'Attention required', val: summary?.attentionRequired ?? 0, color: 'text-orange-400' },
    { label: 'High congestion', val: summary?.highCongestion ?? 0, color: 'text-red-400' },
    { label: 'AI optimizations', val: summary?.optimizations ?? 0, color: 'text-emerald-400' },
    { label: 'Recommendations', val: summary?.recommendations ?? 0, color: 'text-blue-400' },
    { label: 'Predicted congestion', val: summary?.predictedCongestion ?? 0, color: 'text-indigo-400' }
  ];

  return (
    <div className="flex flex-col gap-5 sm:gap-6 h-full min-h-screen text-slate-200 font-sans">

      {/* 1. HEADER */}
      <header className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-3 sm:gap-4 shrink-0 border-b border-slate-800 pb-4">
        <div className="min-w-0">
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white truncate">Traffic Signals</h1>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">Mumbai traffic signal monitoring &amp; AI optimization</p>
        </div>
        <div className="flex flex-wrap items-center gap-2.5 sm:gap-4 shrink-0">
          <span className="px-2.5 py-1.5 bg-blue-900/30 border border-blue-500/60 text-blue-400 text-[10px] sm:text-xs font-semibold uppercase tracking-wide rounded-md flex items-center gap-2 whitespace-nowrap">
            <span className="w-1.5 h-1.5 bg-blue-400 rounded-full motion-safe:animate-pulse" /> Agent active
          </span>
          <div className="flex flex-col items-start sm:items-end">
            <span className="text-[10px] font-semibold text-emerald-500 flex items-center gap-1.5 uppercase tracking-wide whitespace-nowrap">
              <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full" /> Live monitoring
            </span>
            <span className="text-[10px] text-slate-500 font-mono tracking-wide mt-0.5 whitespace-nowrap">
              Updated {lastUpdated ? lastUpdated.toLocaleTimeString() : '--:--:--'}
            </span>
          </div>
        </div>
      </header>

      {/* ERROR STATE */}
      {apiError && (
        <div className="bg-red-900/20 border border-red-500/60 p-3.5 sm:p-4 rounded-xl flex flex-col sm:flex-row gap-3 sm:justify-between sm:items-center shrink-0">
          <span className="text-red-400 font-semibold text-sm">Couldn't load the signal network. Check your connection and try again.</span>
          <button
            onClick={fetchTrafficData}
            className="px-4 py-2 bg-red-600 hover:bg-red-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-red-300 text-white text-xs font-semibold rounded-md uppercase tracking-wide shrink-0"
          >
            Retry
          </button>
        </div>
      )}

      {/* 2. KPI ROW */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 sm:gap-4 shrink-0">
        {kpis.map((kpi, idx) => (
          <div key={idx} className="bg-slate-900 border border-slate-800 p-3.5 sm:p-4 rounded-xl flex flex-col items-center justify-center text-center">
            <span className="text-[10px] text-slate-500 font-semibold tracking-wide text-center leading-tight">{kpi.label}</span>
            <span className={`text-xl sm:text-2xl font-mono font-bold mt-2 ${kpi.color}`}>{kpi.val}</span>
          </div>
        ))}
      </div>

      {/* 3. FILTER BAR */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 sm:p-4 flex flex-col gap-3 shrink-0">
        <div className="flex items-center justify-between gap-3">
          <label htmlFor="signal-search" className="sr-only">Search signal</label>
          <input
            id="signal-search"
            type="text"
            placeholder="Search signal or intersection…"
            value={filters.search}
            onChange={(e) => handleFilterChange('search', e.target.value)}
            className="flex-1 min-w-0 bg-slate-950 border border-slate-700 text-xs sm:text-sm rounded-md px-3 py-2 outline-none text-slate-200 placeholder:text-slate-600 focus:border-blue-500"
          />
          <button
            onClick={() => setFiltersOpen(v => !v)}
            className="lg:hidden shrink-0 px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-xs font-semibold text-slate-300 flex items-center gap-1.5"
            aria-expanded={filtersOpen}
          >
            Filters{activeFilterCount > 0 && <span className="bg-blue-500 text-white rounded-full w-4 h-4 text-[10px] flex items-center justify-center">{activeFilterCount}</span>}
          </button>
        </div>

        <div className={`${filtersOpen ? 'grid' : 'hidden'} lg:grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2.5`}>
          <div className="flex flex-col gap-1">
            <label htmlFor="f-area" className="text-[10px] text-slate-500 font-semibold tracking-wide">Area</label>
            <select id="f-area" value={filters.area} onChange={(e) => handleFilterChange('area', e.target.value)} className="bg-slate-950 border border-slate-700 text-xs rounded-md px-2 py-2 outline-none text-slate-200 focus:border-blue-500">
              <option value="All">All areas</option><option value="Dadar">Dadar</option><option value="BKC">BKC</option><option value="Sion">Sion</option><option value="Wadala">Wadala</option>
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label htmlFor="f-traffic" className="text-[10px] text-slate-500 font-semibold tracking-wide">Traffic</label>
            <select id="f-traffic" value={filters.traffic} onChange={(e) => handleFilterChange('traffic', e.target.value)} className="bg-slate-950 border border-slate-700 text-xs rounded-md px-2 py-2 outline-none text-slate-200 focus:border-blue-500">
              <option value="All">All levels</option><option value="CRITICAL">Critical</option><option value="SEVERE">Severe</option><option value="HIGH">High</option><option value="MODERATE">Moderate</option><option value="LOW">Low</option>
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label htmlFor="f-density" className="text-[10px] text-slate-500 font-semibold tracking-wide">Density</label>
            <select id="f-density" value={filters.density} onChange={(e) => handleFilterChange('density', e.target.value)} className="bg-slate-950 border border-slate-700 text-xs rounded-md px-2 py-2 outline-none text-slate-200 focus:border-blue-500">
              <option value="All">All densities</option><option value="HIGH">High</option><option value="MODERATE">Moderate</option><option value="LOW">Low</option>
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label htmlFor="f-status" className="text-[10px] text-slate-500 font-semibold tracking-wide">Status</label>
            <select id="f-status" value={filters.status} onChange={(e) => handleFilterChange('status', e.target.value)} className="bg-slate-950 border border-slate-700 text-xs rounded-md px-2 py-2 outline-none text-slate-200 focus:border-blue-500">
              <option value="All">All statuses</option><option value="MONITORING">Monitoring</option><option value="OPTIMIZATION RECOMMENDED">Optimization recommended</option>
            </select>
          </div>
          <div className="flex flex-col gap-1 col-span-2 sm:col-span-1">
            <label htmlFor="f-decision" className="text-[10px] text-slate-500 font-semibold tracking-wide">Decision</label>
            <select id="f-decision" value={filters.decision} onChange={(e) => handleFilterChange('decision', e.target.value)} className="bg-slate-950 border border-slate-700 text-xs rounded-md px-2 py-2 outline-none text-slate-200 focus:border-blue-500">
              <option value="All">All decisions</option><option value="Maintain Timing">Maintain timing</option><option value="Increase Green">Increase green</option>
            </select>
          </div>
        </div>

        {activeFilterCount > 0 && (
          <button onClick={clearFilters} className="self-start text-[11px] font-semibold text-slate-500 hover:text-slate-300">Clear filters</button>
        )}
      </div>

      {/* 4. SIGNALS REQUIRING ATTENTION */}
      {attentionSignals.length > 0 && (
        <div className="shrink-0">
          <h2 className="text-xs font-semibold text-slate-400 tracking-wide mb-3">Signals requiring attention</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-6 gap-3 sm:gap-4">
            {attentionSignals.map(sig => (
              <button
                key={sig.signal_code}
                onClick={() => setSelectedSignal(sig)}
                className={`text-left bg-slate-900 border rounded-xl p-3 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-500 ${selectedSignal?.signal_code === sig.signal_code ? 'border-blue-500' : 'border-slate-800 hover:border-slate-600'}`}
              >
                <div className="text-xs font-semibold text-slate-100 truncate">{sig.intersection_name}</div>
                <div className="text-[10px] text-slate-500 font-mono mt-0.5 truncate">{sig.signal_code}</div>
                <div className="mt-3 flex justify-between items-end gap-2">
                  <span className={`text-[10px] font-bold uppercase tracking-wide ${LEVEL_TEXT[sig.traffic_level] || 'text-slate-300'}`}>{sig.traffic_level}</span>
                  <span className="text-[10px] text-blue-400 font-semibold uppercase tracking-wide truncate">{sig.status}</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* MAIN LAYOUT: Table (Left) + Intelligence Panel (Right) */}
      <div
  className={`grid grid-cols-1 xl:grid-cols-12 gap-5 sm:gap-6 items-start transition-all duration-300 ${
    isTableExpanded ? 'xl:grid-cols-12' : ''
  }`}
>

        {/* Left Col: Table */}
        <div
  className={`w-full bg-slate-900 border border-slate-800 rounded-xl flex flex-col overflow-hidden shadow-lg shadow-black/20 transition-all duration-300 ${
    isTableExpanded ? 'xl:col-span-12' : 'xl:col-span-9'
  }`}
>
          <div className="p-3.5 sm:p-4 border-b border-slate-800 bg-slate-800/30 flex flex-col sm:flex-row sm:justify-between sm:items-center gap-3 shrink-0">

  <h3 className="font-semibold text-sm tracking-wide text-slate-200 flex items-center gap-2">
    <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full motion-safe:animate-pulse" />
    Main signal monitoring
  </h3>

  <div className="flex flex-wrap gap-2 items-center text-[11px] sm:text-xs text-slate-400">

    <span className="whitespace-nowrap">
      {totalSignals === 0
        ? 'No signals'
        : `${(page - 1) * limit + 1}–${Math.min(page * limit, totalSignals)} of ${totalSignals}`}
    </span>

    <div className="flex gap-1 sm:ml-2">
      <button
        disabled={page === 1}
        onClick={() => setPage(p => p - 1)}
        className="px-2.5 py-1 bg-slate-800 rounded disabled:opacity-40 hover:bg-slate-700 transition-colors"
      >
        Prev
      </button>

      <button
        disabled={page * limit >= totalSignals}
        onClick={() => setPage(p => p + 1)}
        className="px-2.5 py-1 bg-slate-800 rounded disabled:opacity-40 hover:bg-slate-700 transition-colors"
      >
        Next
      </button>
    </div>

    {/* Expand / Restore */}
    <button
      type="button"
      onClick={() => setIsTableExpanded(prev => !prev)}
      className="ml-1 inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border border-slate-700 bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-white transition-all"
      title={isTableExpanded ? 'Restore intelligence panel' : 'Expand table'}
    >
      {isTableExpanded ? (
        <>
          <svg
            className="w-3.5 h-3.5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M8 3v5H3" />
            <path d="M3 8l5-5" />
            <path d="M16 21v-5h5" />
            <path d="M21 16l-5 5" />
          </svg>
          Restore panel
        </>
      ) : (
        <>
          <svg
            className="w-3.5 h-3.5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M8 3H3v5" />
            <path d="M3 8l5-5" />
            <path d="M16 21h5v-5" />
            <path d="M21 16l-5 5" />
          </svg>
          Expand table
        </>
      )}
    </button>

  </div>
</div>

          {/* Custom Component containing fixed-layout table */}
          <div className="flex-1 min-h-[420px] max-h-[75vh] xl:max-h-[800px] overflow-auto bg-slate-950">
            <SignalMonitoringTable signals={signals} selectedSignalId={selectedSignal?.signal_code} onSelect={setSelectedSignal} />
          </div>
        </div>

        {/* Right Col: Selected Signal Intelligence */}
        {!isTableExpanded && (
        <div className="xl:col-span-3 w-full flex flex-col gap-6">
          {selectedSignal ? (
            <div className="bg-slate-900 border border-slate-700 rounded-xl p-3 sm:p-3 flex flex-col gap-2 max-h-[80vh] xl:max-h-[870px] overflow-y-auto custom-scrollbar shadow-lg shadow-black/20">

              {/* Header Info */}
              <div className="flex justify-between items-start gap-2 border-b border-slate-800 pb-4">
                <div className="min-w-0">
                  <h2 className="text-lg sm:text-xl font-bold text-white truncate">{selectedSignal.intersection_name}</h2>
                  <div className="flex flex-wrap gap-x-2 gap-y-0.5 text-xs text-slate-400 font-mono mt-1">
                    <span>{selectedSignal.signal_code}</span><span aria-hidden="true">•</span><span>{selectedSignal.area}</span>
                  </div>
                </div>
              </div>

              {/* Conditions */}
              <div className="grid grid-cols-3 gap-2 sm:gap-2">
                <div className="bg-slate-950 p-3 rounded-md border border-slate-800">
                  <span className="text-[9px] text-slate-500 tracking-wide font-semibold block mb-1">Current traffic</span>
                  <span className={`text-sm font-bold uppercase ${LEVEL_TEXT[selectedSignal.traffic_level] || 'text-slate-200'}`}>{selectedSignal.traffic_level}</span>
                </div>
                <div className="bg-slate-950 p-3 rounded-md border border-slate-800">
                  <span className="text-[9px] text-slate-500 tracking-wide font-semibold block mb-1">Current density</span>
                  <span className="text-sm font-bold uppercase text-slate-200">{selectedSignal.density_level}</span>
                </div>
                <div className="bg-slate-950 p-3 rounded-md border border-slate-800">
                  <span className="text-[9px] text-slate-500 tracking-wide font-semibold block mb-1">Current phase</span>
                  <span className="text-sm font-bold uppercase text-slate-200">{selectedSignal.current_phase}</span>
                </div>
                <div className="bg-slate-950 p-3 rounded-md border border-slate-800 col-span-3">
                  <span className="text-[9px] text-slate-500 tracking-wide font-semibold block mb-1">Affected approach</span>
                  <span className="text-sm font-bold text-slate-300 break-words">{selectedSignal.affected_approach || 'Network baseline'}</span>
                </div>
              </div>

              {/* Traffic Agent Decision */}
              <div className="bg-blue-900/10 border border-blue-500/30 rounded-md p-3">
                <span className="text-[10px] text-blue-400 tracking-wide font-semibold block mb-1">Traffic agent decision</span>
                <div className="text-base text-slate-100 font-bold break-words">{selectedSignal.ai_decision}</div>
              </div>

              {/* Signal Optimization Visual */}
              <div className="bg-slate-950 p-3 rounded-md border border-slate-800 flex flex-col gap-3">
                <h3 className="text-[10px] font-semibold text-slate-400 tracking-wide border-b border-slate-800 pb-2">Signal optimization impact</h3>

                <div className="flex flex-col gap-2">
                  <div className="flex flex-wrap justify-between items-end gap-1">
                    <span className="text-[10px] font-semibold text-slate-500 tracking-wide">Current — cycle {selectedSignal.cycle_time_seconds}s</span>
                    <span className="text-xs font-mono text-slate-400">G {selectedSignal.current_green_seconds}s · R {selectedSignal.current_red_seconds}s</span>
                  </div>
                  {renderTimingBar(selectedSignal.current_green_seconds, selectedSignal.current_red_seconds)}
                </div>

                <div className="flex items-center gap-3 text-slate-600">
                  <div className="h-px flex-1 bg-slate-800" />
                  <span className="text-[10px] font-semibold tracking-wide">Traffic Agent recommendation</span>
                  <div className="h-px flex-1 bg-slate-800" />
                </div>

                <div className="flex flex-col gap-2">
                  <div className="flex flex-wrap justify-between items-end gap-1">
                    <span className="text-[10px] font-semibold text-emerald-400 tracking-wide">Recommended</span>
                    <span className="text-xs font-mono text-emerald-300">G {selectedSignal.recommended_green_seconds ?? selectedSignal.current_green_seconds}s · R {selectedSignal.recommended_red_seconds ?? selectedSignal.current_red_seconds}s</span>
                  </div>
                  {renderTimingBar(selectedSignal.recommended_green_seconds ?? selectedSignal.current_green_seconds, selectedSignal.recommended_red_seconds ?? selectedSignal.current_red_seconds)}
                </div>

                {!!selectedSignal.green_adjustment_seconds && (
                  <div className="mt-1 text-xs font-mono font-semibold flex flex-wrap justify-center gap-4 border-t border-slate-800 pt-3">
                    <span className="text-emerald-400">Green {selectedSignal.green_adjustment_seconds > 0 ? '+' : ''}{selectedSignal.green_adjustment_seconds}s</span>
                    <span className="text-red-400">Red {selectedSignal.red_adjustment_seconds > 0 ? '+' : ''}{selectedSignal.red_adjustment_seconds}s</span>
                  </div>
                )}
              </div>

              {/* Forecast & Reason */}
              <div className="grid grid-cols-1 gap-2.5 sm:gap-3">
                <div className="bg-slate-950 p-3 rounded-md border border-slate-800">
                  <span className="text-[10px] text-slate-500 tracking-wide font-semibold block mb-1">Forecast — next {selectedSignal.prediction_horizon_minutes}m</span>
                  <span className="text-sm font-bold text-orange-400 uppercase">{selectedSignal.predicted_traffic_level || 'N/A'}</span>
                </div>
                <div className="bg-slate-950 p-3 rounded-md border border-slate-800">
                  <span className="text-[10px] text-slate-500 tracking-wide font-semibold block mb-1">Reason</span>
                  <p className="text-xs text-slate-300 italic break-words">"{selectedSignal.reason || 'Sustaining optimal throughput parameters.'}"</p>
                </div>
              </div>

              {/* Expected Impact */}
              <div className="bg-emerald-950/20 p-4 rounded-md border border-emerald-900/50">
                <span className="text-[10px] text-emerald-500 tracking-wide font-semibold block mb-1">Expected impact</span>
                <p className="text-sm font-bold text-emerald-400 break-words">{selectedSignal.expected_impact || 'Maintain intersection throughput'}</p>
                <div className="mt-3 flex flex-wrap justify-between gap-2 border-t border-emerald-900/50 pt-2">
                  <span className="text-[9px] text-slate-500 font-mono tracking-wide">Confidence {selectedSignal.confidence ?? '92'}%</span>
                  <span className="text-[9px] text-slate-500 font-mono tracking-wide">Updated {selectedSignal.last_observed_at ? new Date(selectedSignal.last_observed_at).toLocaleTimeString() : '--:--:--'}</span>
                </div>
              </div>

            </div>
          ) : (
            <div className="bg-slate-900 border border-slate-700 rounded-xl p-5 flex items-center justify-center text-center text-slate-500 italic min-h-[200px] xl:h-full">
              Select a signal from the table to view intelligence &amp; optimization impact.
            </div>
          )}
        </div>
        )}
      </div>

      {/* LOWER SECTION: Areas, Active Conditions, Logs */}
      <div className="flex flex-col gap-5 sm:gap-6 shrink-0 border-t border-slate-800 pt-5 sm:pt-6">

        {/* Mumbai Signal Network By Area — compact table instead of one big card per area */}
        {/* <div className="flex flex-col gap-3">
          <div className="flex items-baseline justify-between gap-3">
            <h2 className="text-sm font-semibold text-slate-400 tracking-wide">Mumbai signal network by area</h2>
            <span className="text-[11px] text-slate-500 font-mono whitespace-nowrap">
              {areaStats.length} areas · {areaStats.reduce((sum, a) => sum + (a.signalCount || 0), 0)} signals
            </span>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
            <div className="overflow-x-auto custom-scrollbar">
              <table className="w-full min-w-[640px] text-left text-xs">
                <thead className="bg-slate-950 text-[10px] uppercase tracking-wide text-slate-500 border-b border-slate-800">
                  <tr>
                    <th className="px-4 py-2.5 font-semibold">Area</th>
                    <th className="px-3 py-2.5 font-semibold text-right">Signals</th>
                    <th className="px-3 py-2.5 font-semibold text-right">Low</th>
                    <th className="px-3 py-2.5 font-semibold text-right">Mod</th>
                    <th className="px-3 py-2.5 font-semibold text-right">High</th>
                    <th className="px-3 py-2.5 font-semibold text-right">Sev</th>
                    <th className="px-3 py-2.5 font-semibold text-right">Crit</th>
                    <th className="px-3 py-2.5 font-semibold text-right">AI recs</th>
                    <th className="px-4 py-2.5 font-semibold text-right">Review</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {areaStats.length === 0 ? (
                    <tr><td colSpan={9} className="px-4 py-8 text-center text-slate-500 italic">No area data available.</td></tr>
                  ) : areaStats.map((area, idx) => {
                    const isActive = filters.area === area.area;
                    return (
                      <tr
                        key={idx}
                        onClick={() => handleFilterChange('area', area.area)}
                        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleFilterChange('area', area.area); } }}
                        tabIndex={0}
                        role="button"
                        aria-pressed={isActive}
                        className={`cursor-pointer outline-none transition-colors ${isActive ? 'bg-blue-900/20' : 'hover:bg-slate-800/50'}`}
                      >
                        <td className={`px-4 py-2.5 font-semibold truncate max-w-[160px] border-l-2 ${isActive ? 'border-blue-500 text-blue-400' : 'border-transparent text-slate-200'}`}>
                          {area.area}
                        </td>
                        <td className="px-3 py-2.5 text-right font-mono text-slate-300">{area.signalCount ?? 0}</td>
                        <td className="px-3 py-2.5 text-right font-mono">{renderCount(area.lowTraffic, 'text-emerald-400')}</td>
                        <td className="px-3 py-2.5 text-right font-mono">{renderCount(area.moderateTraffic, 'text-yellow-400')}</td>
                        <td className="px-3 py-2.5 text-right font-mono">{renderCount(area.highTraffic, 'text-orange-400')}</td>
                        <td className="px-3 py-2.5 text-right font-mono">{renderCount(area.severeTraffic, 'text-red-400')}</td>
                        <td className="px-3 py-2.5 text-right font-mono">{renderCount(area.criticalTraffic, 'text-red-600')}</td>
                        <td className="px-3 py-2.5 text-right font-mono">{renderCount(area.optimizations, 'text-blue-400')}</td>
                        <td className="px-4 py-2.5 text-right font-mono">{renderCount(area.attentionRequired, 'text-orange-400')}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div> */}

        {/* Active Traffic Conditions & Agent Activity — side by side instead of stacked */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 sm:gap-6 mb-5">

          <div className="flex flex-col gap-3">
            <h2 className="text-sm font-semibold text-slate-400 tracking-wide">Active traffic conditions</h2>
            <div className="flex flex-col gap-3">
              {activeConditions.length === 0 ? (
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 text-slate-500 text-sm italic">No severe or critical conditions detected.</div>
              ) : activeConditions.map((cond, i) => (
                <div key={i} className="bg-slate-900 border-l-4 border-red-500 p-4 rounded-r-xl border-y border-r border-slate-800">
                  <div className="flex flex-wrap justify-between items-start gap-2 mb-2">
                    <h4 className="font-semibold text-slate-100 text-sm">{cond.intersection_name}</h4>
                    <span className="text-[10px] font-bold uppercase tracking-wide text-red-400">{cond.traffic_level}</span>
                  </div>
                  <div className="text-xs text-slate-300 font-medium mb-1">Agent action: <span className="text-blue-400">{cond.ai_decision}</span></div>
                  <div className="text-xs font-mono text-slate-500 mb-2 break-words">Current G {cond.current_green_seconds}s / R {cond.current_red_seconds}s → Rec G {cond.recommended_green_seconds}s / R {cond.recommended_red_seconds}s</div>
                  <div className="flex flex-wrap justify-between items-center gap-2 text-[10px] font-semibold tracking-wide border-t border-slate-800 pt-2">
                    <span className="text-orange-400">Forecast {cond.predicted_traffic_level}</span>
                    <span className="text-emerald-400 truncate max-w-full" title={cond.expected_impact}>Impact: {cond.expected_impact}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="flex flex-col gap-3">
            <h2 className="text-sm font-semibold text-slate-400 tracking-wide">Traffic agent activity</h2>
            <div className="bg-black border border-slate-800 rounded-xl p-4 h-[550px] overflow-y-auto custom-scrollbar">
              <div className="font-mono text-[11px] leading-relaxed space-y-2">
                {activityFeed.length === 0 ? (
                  <div className="text-slate-600 italic">No recent activity logs available.</div>
                ) : (
                  activityFeed.map((log, i) => (
                    <div key={i} className="pb-2 border-b border-slate-900/50 last:border-0 last:pb-0 break-words">
                      <span className="text-slate-500">{new Date(log.timestamp).toLocaleTimeString()}</span>{' '}
                      <span className="text-blue-400">[{log.area}]</span>{' '}
                      <span className="text-slate-300 font-semibold">{log.signal}:</span>{' '}
                      <span className={['CRITICAL', 'SEVERE'].includes(log.trafficCondition) ? 'text-red-400' : 'text-orange-400'}>{log.trafficCondition} traffic detected.</span>{' '}
                      <span className="text-slate-400">Decision: {log.decision}.</span>{' '}
                      <span className="text-emerald-400 font-semibold">({log.recommendation})</span>{' '}
                      <span className="text-slate-500">Forecast: {log.prediction}.</span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}