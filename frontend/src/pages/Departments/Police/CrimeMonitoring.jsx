import React, { useState, useEffect, useMemo, useRef, useCallback, memo } from 'react';
import { useNavigate } from 'react-router-dom';
import { usePoliceStore } from '../../../store/usePoliceStore';
import { useMapStore } from '../../../store/useMapStore';

// Change this if your video server lives elsewhere.
const VIDEO_BASE = 'http://localhost:5000/videos';

// ==========================================
// ICONS
// ==========================================
const Svg = ({ d, className = 'w-4 h-4' }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d={d} />
  </svg>
);
const SearchIcon = () => <Svg className="w-4 h-4 text-slate-400" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />;
const MaximizeIcon = () => <Svg d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />;
const CloseIcon = () => <Svg className="w-5 h-5" d="M6 18L18 6M6 6l12 12" />;
const ActivityIcon = () => <Svg d="M13 10V3L4 14h7v7l9-11h-7z" />;
const ShieldIcon = () => <Svg d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />;
const CameraIcon = () => <Svg d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />;

// ==========================================
// UTILS & ADAPTERS
// ==========================================
const getRiskLabel = (score) => (score >= 80 ? 'CRITICAL' : score >= 60 ? 'HIGH' : score >= 30 ? 'MEDIUM' : 'LOW');

const RISK_STYLES = {
  CRITICAL: { text: 'text-red-400', badge: 'text-red-400 border-red-500/30 bg-red-500/10' },
  HIGH: { text: 'text-orange-400', badge: 'text-orange-400 border-orange-500/30 bg-orange-500/10' },
  MEDIUM: { text: 'text-amber-400', badge: 'text-amber-400 border-amber-500/30 bg-amber-500/10' },
  LOW: { text: 'text-emerald-400', badge: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10' },
};
const riskStyle = (level) => RISK_STYLES[level] || { text: 'text-slate-400', badge: 'text-slate-400 border-slate-500/30 bg-slate-500/10' };

const detHash = (str) => {
  let hash = 0;
  const s = String(str);
  for (let i = 0; i < s.length; i++) hash = ((hash << 5) - hash + s.charCodeAt(i)) | 0;
  return Math.abs(hash);
};

const crimeId = (c) => (c ? c.id ?? c.incident_id ?? null : null);
const crimeLabel = (c) => (c ? c.category || c.incident_type || 'Incident' : '');

const DEFAULT_LOCATIONS = ['Andheri East', 'Bandra West', 'Kurla Junction', 'Dadar', 'Sion Circle', 'Matunga', 'Parel', 'Worli Sea Face'];
const RISK_FILTERS = ['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];

const buildCrimeCameraIntelligence = (entities = [], crimes = [], activeIncidents = [], baseTime) => {
  const cctvs = (entities || []).filter((e) => e.entity_type === 'cctv');
  const baseCameras = cctvs.length > 0 ? cctvs : Array.from({ length: 24 }).map((_, i) => ({
    entity_id: `CAM-MUM-${String(i + 1).padStart(3, '0')}`,
    name: `Surveillance Node ${i + 1}`,
    location_name: DEFAULT_LOCATIONS[i % DEFAULT_LOCATIONS.length],
  }));

  const allCrimes = [...(crimes || []), ...(activeIncidents || [])];
  const videoMap = ['1.mp4', '2.mp4', '3.mp4', '4.mp4', '5.mp4'];

  return baseCameras
    .map((cam, idx) => {
      const isOffline = idx % 11 === 0;
      const correlatedCrime = allCrimes.length > 0 ? allCrimes[detHash(cam.entity_id) % allCrimes.length] : null;

      let baseRisk = 15;
      let detection = 'Nominal monitoring';
      let confidence = 0;
      let recommendation = 'Continue routine surveillance';
      let activeTags = [];

      if (!isOffline && correlatedCrime) {
        const sev = String(correlatedCrime.severity || '').toUpperCase();
        if (sev === 'CRITICAL') baseRisk += 65;
        else if (sev === 'HIGH') baseRisk += 50;
        else if (sev === 'MEDIUM') baseRisk += 25;

        const t = new Date(correlatedCrime.time || correlatedCrime.reported_at || correlatedCrime.created_at).getTime();
        if (!Number.isNaN(t) && baseTime - t < 86400000) baseRisk += 15;

        confidence = 75 + (detHash(`${cam.entity_id}${crimeId(correlatedCrime)}`) % 20);

        if (baseRisk >= 80) {
          detection = 'Potential violent incident / Weapon drawn';
          recommendation = 'Immediate officer verification. Dispatch rapid response.';
          activeTags = ['PERSON DETECTED', 'UNUSUAL MOVEMENT', 'HIGH THREAT'];
        } else if (baseRisk >= 60) {
          detection = 'Suspicious movement / Unauthorized access';
          recommendation = 'Verify with patrol unit. Correlate with active investigations.';
          activeTags = ['PERSON DETECTED', 'RESTRICTED ZONE'];
        } else if (baseRisk >= 30) {
          detection = 'Unusual crowd formation';
          recommendation = 'Monitor density. Alert traffic units if spillover occurs.';
          activeTags = ['CROWD DETECTED', 'DENSITY ANOMALY'];
        } else {
          detection = 'Routine tracking';
          activeTags = ['PERSON DETECTED', 'VEHICLE DETECTED'];
        }
      }

      const riskScore = isOffline ? 0 : Math.min(baseRisk, 99);
      return {
        id: cam.entity_id,
        location: cam.location_name || cam.name || cam.entity_id,
        direction: ['Northbound', 'Southbound', 'Eastbound', 'Westbound'][idx % 4],
        status: isOffline ? 'OFFLINE' : 'ONLINE',
        riskScore,
        riskLevel: isOffline ? 'LOW' : getRiskLabel(riskScore),
        detection: isOffline ? 'Camera offline' : detection,
        confidence,
        recommendation: isOffline ? 'Dispatch technician to restore feed' : recommendation,
        activeTags,
        streamUrl: isOffline ? null : `${VIDEO_BASE}/${videoMap[idx % videoMap.length]}`,
        correlatedCrime: isOffline ? null : correlatedCrime,
        timestamp: new Date(baseTime - (detHash(cam.entity_id) % 3600000)).toLocaleTimeString(),
      };
    })
    .sort((a, b) => b.riskScore - a.riskScore || a.id.localeCompare(b.id));
};

// ==========================================
// SMALL STABLE COMPONENTS (defined outside the page so they never remount)
// ==========================================
const Clock = memo(function Clock() {
  const [time, setTime] = useState(() => new Date().toLocaleTimeString());
  useEffect(() => {
    const t = setInterval(() => setTime(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(t);
  }, []);
  return <span className="tabular-nums">{time}</span>;
});

const RiskBadge = ({ level }) => (
  <span className={`text-[10px] px-1.5 py-0.5 rounded border uppercase font-bold tracking-wider shrink-0 ${riskStyle(level).badge}`}>{level}</span>
);

const Panel = ({ title, icon, className = '', children }) => (
  <section className={`bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col min-h-0 ${className}`}>
    <h3 className="text-xs font-semibold text-slate-400 mb-3 shrink-0 flex items-center gap-2">{icon}{title}</h3>
    {children}
  </section>
);

const FeedOverlay = memo(function FeedOverlay({ camera }) {
  if (!camera || camera.status === 'OFFLINE') return null;
  return (
    <div className="absolute inset-0 pointer-events-none p-3 sm:p-4 flex flex-col justify-between">
      <div className="flex justify-between items-start gap-2 mt-8">
        <div className="bg-black/60 backdrop-blur-sm px-2.5 py-1.5 rounded border border-slate-700/50 flex flex-col gap-1">
          <span className="text-[10px] font-semibold text-cyan-400 flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-500 motion-safe:animate-pulse" />
            AI analysis active
          </span>
          {camera.activeTags.map((tag) => (
            <span key={tag} className="text-[10px] font-mono text-slate-200">{tag}</span>
          ))}
        </div>
        <div className="bg-black/60 backdrop-blur-sm px-2.5 py-1.5 rounded border border-slate-700/50 text-right">
          <span className="text-[10px] font-semibold text-slate-400 block">Risk score</span>
          <span className={`text-lg font-mono font-bold ${riskStyle(camera.riskLevel).text}`}>{camera.riskScore}</span>
        </div>
      </div>
      {camera.riskScore >= 60 && (
        <div className="self-center mb-2 max-w-full border border-red-500/50 bg-red-500/10 px-3 py-1 rounded backdrop-blur text-center">
          <span className="text-xs font-bold text-red-400 motion-safe:animate-pulse">{camera.detection}</span>
        </div>
      )}
    </div>
  );
});

// Video with loading + error states. Re-keyed by the caller so switching cameras resets it.
const VideoFeed = ({ camera, fit = 'cover' }) => {
  const [state, setState] = useState('loading'); // loading | ready | error
  if (camera.status !== 'ONLINE' || !camera.streamUrl || state === 'error') {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center gap-2 text-slate-500 text-sm bg-slate-950">
        <CameraIcon />
        <span className="font-semibold">{camera.status === 'OFFLINE' ? 'Camera offline' : 'Feed unavailable'}</span>
        <span className="text-xs text-slate-600 text-center px-6">
          {camera.status === 'OFFLINE' ? 'A technician has been recommended for this node.' : 'The video server could not be reached.'}
        </span>
      </div>
    );
  }
  return (
    <>
      {state === 'loading' && (
        <div className="absolute inset-0 flex items-center justify-center bg-black text-xs text-slate-500">Connecting to feed…</div>
      )}
      <video
        src={camera.streamUrl}
        autoPlay loop muted playsInline preload="auto"
        onCanPlay={() => setState('ready')}
        onError={() => setState('error')}
        className={`w-full h-full transition-opacity duration-300 ${fit === 'contain' ? 'object-contain' : 'object-cover'} ${state === 'ready' ? 'opacity-100' : 'opacity-0'}`}
      />
    </>
  );
};

const Label = ({ children }) => <span className="text-[11px] text-slate-500 font-semibold block mb-1">{children}</span>;

const FullscreenModal = ({ camera, onClose, onOpenCase }) => {
  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose();
    document.addEventListener('keydown', onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = prev;
    };
  }, [onClose]);

  const caseId = crimeId(camera.correlatedCrime);
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-2 sm:p-4 bg-slate-950/90 backdrop-blur-md" onClick={onClose}>
      <div role="dialog" aria-modal="true" aria-label={`Camera ${camera.id}`} className="bg-slate-900 border border-slate-700 w-full max-w-7xl h-[92vh] sm:h-[85vh] rounded-xl flex flex-col overflow-hidden shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="p-4 border-b border-slate-800 flex justify-between items-center bg-slate-950 shrink-0">
          <h2 className="text-base sm:text-lg font-bold text-slate-100 flex items-center gap-3 min-w-0">
            <CameraIcon /> {camera.id}
            <span className="text-slate-500 font-mono text-xs truncate">| {camera.location}</span>
          </h2>
          <button onClick={onClose} aria-label="Close (Esc)" title="Close (Esc)" className="text-slate-400 hover:text-white transition-colors p-1 rounded focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-500"><CloseIcon /></button>
        </div>
        <div className="flex-1 flex flex-col lg:flex-row min-h-0">
          <div className="lg:w-2/3 h-1/2 lg:h-auto bg-black relative border-b lg:border-b-0 lg:border-r border-slate-800 shrink-0 lg:shrink">
            <VideoFeed key={camera.id} camera={camera} fit="contain" />
            <FeedOverlay camera={camera} />
          </div>
          <div className="lg:w-1/3 flex-1 bg-slate-900 p-5 sm:p-6 overflow-y-auto space-y-6 min-h-0">
            <h3 className="text-sm font-semibold text-slate-300">AI crime agent assessment</h3>
            <div><Label>Detection</Label><span className="text-sm font-bold text-slate-200">{camera.detection}</span></div>
            <div className="flex gap-6">
              <div><Label>Confidence</Label><span className="text-lg font-mono text-cyan-400 font-bold">{camera.confidence}%</span></div>
              <div><Label>Risk</Label><span className={`text-lg font-mono font-bold ${riskStyle(camera.riskLevel).text}`}>{camera.riskLevel}</span></div>
            </div>
            <div><Label>Recommendation</Label><p className="text-sm text-slate-300 bg-slate-950 border border-slate-800 p-3 rounded">{camera.recommendation}</p></div>
            {camera.correlatedCrime && (
              <div className="border-t border-slate-800 pt-4">
                <Label>Correlated investigation</Label>
                <div className="bg-indigo-950/20 border border-indigo-900/40 p-3 rounded">
                  <div className="flex justify-between items-start mb-1 gap-2">
                    <span className="text-xs font-bold text-indigo-400">{crimeLabel(camera.correlatedCrime)}</span>
                    <span className="text-[10px] font-mono text-slate-500">{caseId}</span>
                  </div>
                  <p className="text-xs text-slate-400 leading-relaxed mb-3">AI correlates this detection with an active case file within spatial and temporal bounds.</p>
                  <button onClick={() => onOpenCase(caseId)} className="w-full bg-indigo-900/30 hover:bg-indigo-900/50 text-indigo-300 text-xs font-semibold py-2 rounded transition-colors border border-indigo-800/50">Open case file</button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// ==========================================
// MAIN COMPONENT
// ==========================================
export default function CrimeMonitoring() {
  const navigate = useNavigate();
  const { crimes, activeIncidents, fetchInitialData } = usePoliceStore();
  const { entities, initializeMap } = useMapStore();

  const [searchQuery, setSearchQuery] = useState('');
  const [filterRisk, setFilterRisk] = useState('ALL');
  const [selectedId, setSelectedId] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const baseTime = useRef(Date.now()).current; // keeps timestamps stable across data refreshes

  // Load data once on mount (previously re-ran whenever entities changed)
  useEffect(() => {
    fetchInitialData();
    if (!entities || entities.length === 0) {
      fetch('/api/map/data')
        .then((res) => res.json())
        .then((data) => initializeMap(data))
        .catch(() => {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Derived data — no extra state, no extra render pass
  const cctvData = useMemo(
    () => buildCrimeCameraIntelligence(entities, crimes, activeIncidents, baseTime),
    [entities, crimes, activeIncidents, baseTime]
  );

  const selectedCamera = useMemo(
    () => cctvData.find((c) => c.id === selectedId) || cctvData[0] || null,
    [cctvData, selectedId]
  );

  const riskCounts = useMemo(() => {
    const counts = { ALL: cctvData.length, CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 };
    cctvData.forEach((c) => { counts[c.riskLevel] += 1; });
    return counts;
  }, [cctvData]);

  const filteredCameras = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return cctvData.filter((cam) => {
      const matchSearch = !q || cam.id.toLowerCase().includes(q) || cam.location.toLowerCase().includes(q);
      return matchSearch && (filterRisk === 'ALL' || cam.riskLevel === filterRisk);
    });
  }, [cctvData, searchQuery, filterRisk]);

  const kpis = useMemo(() => {
    const highRisk = cctvData.filter((c) => c.riskLevel === 'HIGH' || c.riskLevel === 'CRITICAL').length;
    return {
      monitored: cctvData.length,
      online: cctvData.filter((c) => c.status === 'ONLINE').length,
      detections: cctvData.filter((c) => c.riskScore >= 30).length,
      highRisk,
      activeCrimes: (activeIncidents || []).length,
      decisionsPending: highRisk,
    };
  }, [cctvData, activeIncidents]);

  const onlinePct = kpis.monitored ? Math.round((kpis.online / kpis.monitored) * 100) : 0;

  const cityThreat = useMemo(() => {
    if (kpis.highRisk > 4) return { level: 'CRITICAL', color: 'text-red-400' };
    if (kpis.highRisk > 1) return { level: 'ELEVATED', color: 'text-orange-400' };
    return { level: 'NOMINAL', color: 'text-emerald-400' };
  }, [kpis.highRisk]);

  const pending = useMemo(() => cctvData.filter((c) => c.riskScore >= 60), [cctvData]);
  const recentDetections = useMemo(() => cctvData.filter((c) => c.riskScore > 0).slice(0, 6), [cctvData]);

  const closeFullscreen = useCallback(() => setIsFullscreen(false), []);
  const openCase = useCallback((id) => {
    setIsFullscreen(false);
    if (id != null) navigate(`/police/crimes?id=${id}`);
  }, [navigate]);
  const resetFilters = () => { setSearchQuery(''); setFilterRisk('ALL'); };

  const kpiCards = [
    { label: 'Cameras monitored', val: kpis.monitored },
    { label: 'Cameras online', val: kpis.online, color: 'text-emerald-400' },
    { label: 'AI detections', val: kpis.detections, color: 'text-cyan-400' },
    { label: 'High-risk detections', val: kpis.highRisk, color: 'text-orange-400', onClick: () => setFilterRisk('HIGH') },
    { label: 'Active crimes', val: kpis.activeCrimes, color: 'text-indigo-400' },
    { label: 'Decisions pending', val: kpis.decisionsPending, color: 'text-red-400' },
  ];

  const focusRing = 'focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-500';

  return (
    <div className="flex flex-col gap-6 text-slate-100 font-sans p-4 sm:p-6 min-h-screen bg-slate-950 overflow-x-hidden">
      {/* HEADER */}
      <header className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">AI crime surveillance</h1>
          <p className="text-sm text-slate-400 mt-1">City-wide CCTV intelligence and automated crime risk assessment.</p>
        </div>
        <div className="md:text-right flex flex-col md:items-end gap-1.5">
          <div className="text-xs text-cyan-400 font-semibold flex items-center gap-1.5 bg-cyan-950/30 border border-cyan-900/50 px-2.5 py-1 rounded-full">
            <span className="w-2 h-2 rounded-full bg-cyan-500 motion-safe:animate-pulse" /> AI agent active
          </div>
          <span className="text-xs text-slate-500 font-mono">Last analysis: <Clock /></span>
        </div>
      </header>

      {/* KPI STRIP */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
        {kpiCards.map((k) => {
          const Tag = k.onClick ? 'button' : 'div';
          return (
            <Tag key={k.label} onClick={k.onClick} className={`text-left bg-slate-900/80 border border-slate-800 px-4 py-3 rounded-xl flex flex-col ${k.onClick ? `hover:border-slate-600 transition-colors ${focusRing}` : ''}`}>
              <span className="text-[11px] font-semibold text-slate-500 mb-1">{k.label}</span>
              <span className={`text-2xl font-mono font-bold tabular-nums ${k.color || 'text-slate-200'}`}>{k.val}</span>
            </Tag>
          );
        })}
      </div>

      {/* FILTER BAR */}
      <div className="bg-slate-900/50 border border-slate-800 p-3 rounded-xl flex flex-col lg:flex-row gap-3 lg:items-center">
        <div className="relative w-full lg:w-72">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"><SearchIcon /></div>
          <input
            type="text" aria-label="Search cameras" placeholder="Search by camera ID or location"
            value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-slate-950 border border-slate-700 text-sm rounded-lg pl-10 pr-8 py-2 text-slate-200 outline-none focus:border-cyan-500 transition-colors"
          />
          {searchQuery && (
            <button onClick={() => setSearchQuery('')} aria-label="Clear search" className={`absolute inset-y-0 right-2 text-slate-500 hover:text-slate-200 text-lg leading-none ${focusRing}`}>×</button>
          )}
        </div>
        <div className="flex flex-wrap gap-2" role="group" aria-label="Filter by risk level">
          {RISK_FILTERS.map((r) => (
            <button
              key={r} aria-pressed={filterRisk === r} onClick={() => setFilterRisk(r)}
              className={`text-xs font-semibold px-3 py-1.5 rounded-full border transition-colors ${focusRing} ${
                filterRisk === r ? 'bg-cyan-900/30 border-cyan-500/50 text-cyan-300' : 'bg-slate-950 border-slate-700 text-slate-400 hover:border-slate-500'
              }`}
            >
              {r === 'ALL' ? 'All' : r[0] + r.slice(1).toLowerCase()} <span className="font-mono opacity-70">{riskCounts[r]}</span>
            </button>
          ))}
        </div>
      </div>

      {/* MAIN WORKSPACE */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 xl:h-[620px]">
        {/* Camera list */}
        <section className="xl:col-span-3 bg-slate-900/50 border border-slate-800 rounded-xl p-3 flex flex-col max-h-[360px] xl:max-h-none min-h-0">
          <h3 className="text-xs font-semibold text-slate-400 mb-3 px-1 flex justify-between">
            <span>Surveillance nodes</span>
            <span className="font-mono text-slate-500">{filteredCameras.length}/{cctvData.length}</span>
          </h3>
          <div className="flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-2 pr-1 min-h-0">
            {filteredCameras.length === 0 ? (
              <div className="text-center text-slate-500 text-sm mt-6 flex flex-col items-center gap-3">
                No cameras match your filters.
                <button onClick={resetFilters} className={`text-xs font-semibold text-cyan-400 hover:text-cyan-300 underline ${focusRing}`}>Clear filters</button>
              </div>
            ) : (
              filteredCameras.map((cam) => {
                const active = selectedCamera?.id === cam.id;
                const offline = cam.status === 'OFFLINE';
                return (
                  <button
                    key={cam.id} onClick={() => setSelectedId(cam.id)} aria-pressed={active}
                    className={`text-left p-3 rounded-lg border transition-colors duration-150 shrink-0 ${focusRing} ${
                      active ? 'bg-cyan-900/20 border-cyan-500/50' : 'bg-slate-950 border-slate-800 hover:border-slate-600'
                    }`}
                  >
                    <div className="flex justify-between items-start mb-1.5 gap-2">
                      <span className={`text-xs font-mono font-bold ${offline ? 'text-slate-600' : 'text-slate-300'}`}>{cam.id}</span>
                      {offline ? <span className="text-[10px] px-1.5 py-0.5 rounded border border-slate-700 text-slate-500 font-bold">OFFLINE</span> : <RiskBadge level={cam.riskLevel} />}
                    </div>
                    <div className={`text-sm font-semibold truncate ${offline ? 'text-slate-600' : 'text-slate-200'}`}>{cam.location}</div>
                    <div className="text-[11px] text-slate-500 mt-1 truncate">{cam.detection}</div>
                  </button>
                );
              })
            )}
          </div>
        </section>

        {/* Live feed */}
        <section className="xl:col-span-6 bg-slate-900 border border-slate-700/80 shadow-lg rounded-xl overflow-hidden relative aspect-video xl:aspect-auto min-h-[260px]">
          {selectedCamera ? (
            <>
              <div className="absolute top-0 left-0 right-0 p-3 bg-gradient-to-b from-slate-950/80 to-transparent z-10 flex justify-between items-start pointer-events-none">
                <div className="min-w-0">
                  <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2"><CameraIcon /> {selectedCamera.id}</h3>
                  <span className="text-[11px] font-mono text-slate-300 bg-black/50 px-1.5 rounded mt-1 inline-block backdrop-blur">{selectedCamera.location} • {selectedCamera.direction}</span>
                </div>
                <button onClick={() => setIsFullscreen(true)} aria-label="Expand feed" title="Expand feed" className={`pointer-events-auto p-1.5 bg-black/50 hover:bg-black/80 rounded border border-slate-700 text-slate-300 transition-colors backdrop-blur ${focusRing}`}><MaximizeIcon /></button>
              </div>
              <div className="absolute inset-0 bg-black">
                <VideoFeed key={selectedCamera.id} camera={selectedCamera} />
                <FeedOverlay camera={selectedCamera} />
              </div>
            </>
          ) : (
            <div className="w-full h-full flex items-center justify-center text-slate-500 text-sm">No cameras available.</div>
          )}
        </section>

        {/* AI intelligence panel */}
        <section className="xl:col-span-3 bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col gap-5 overflow-y-auto custom-scrollbar min-h-0">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-3 shrink-0">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-500 motion-safe:animate-pulse" />
            <h2 className="text-sm font-semibold text-cyan-400">AI crime agent</h2>
          </div>
          {selectedCamera ? (
            <div className="flex flex-col gap-5 flex-1">
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-slate-950 border border-slate-800 p-3 rounded text-center">
                  <Label>Risk</Label>
                  <span className={`text-lg font-mono font-bold ${riskStyle(selectedCamera.riskLevel).text}`}>{selectedCamera.riskLevel}</span>
                </div>
                <div className="bg-slate-950 border border-slate-800 p-3 rounded text-center">
                  <Label>Confidence</Label>
                  <span className="text-lg font-mono font-bold text-cyan-400">{selectedCamera.status === 'OFFLINE' ? '—' : `${selectedCamera.confidence}%`}</span>
                </div>
              </div>
              <div>
                <Label>Current detection</Label>
                <span className="text-sm font-bold text-slate-200">{selectedCamera.detection}</span>
                <span className="text-xs text-slate-400 font-mono block mt-1">Detected: {selectedCamera.timestamp}</span>
              </div>
              <div>
                <Label>AI reasoning</Label>
                <p className="text-xs text-slate-300 leading-relaxed bg-slate-950 p-2.5 border border-slate-800 rounded">
                  {selectedCamera.status === 'OFFLINE'
                    ? 'No video available, so no analysis can run on this node.'
                    : selectedCamera.riskScore >= 60
                    ? 'Unusual movement pattern detected near a known high-crime corridor. Activity cross-references with the spatial risk model.'
                    : 'Routine monitoring active. No critical deviations from baseline environmental models.'}
                </p>
              </div>
              <div>
                <Label>Recommended action</Label>
                <p className="text-xs leading-relaxed bg-cyan-950/20 border border-cyan-900/30 p-2.5 rounded text-cyan-100">{selectedCamera.recommendation}</p>
              </div>
              <div className="mt-auto pt-2 flex flex-col gap-2">
                <button onClick={() => navigate('/police/agent')} className={`w-full bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold py-2 rounded transition-colors ${focusRing}`}>Review agent decision</button>
                {selectedCamera.correlatedCrime && (
                  <button onClick={() => openCase(crimeId(selectedCamera.correlatedCrime))} className={`w-full bg-indigo-900/40 hover:bg-indigo-900/60 border border-indigo-500/30 text-indigo-300 text-xs font-semibold py-2 rounded transition-colors ${focusRing}`}>Open crime record</button>
                )}
              </div>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center text-sm text-slate-500 text-center">Select a camera to see the AI analysis.</div>
          )}
        </section>
      </div>

      {/* SECONDARY ROW */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Panel title="AI decision queue" icon={<ActivityIcon />} className="h-[380px]">
          <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 space-y-3 min-h-0">
            {pending.length === 0 ? (
              <div className="text-sm text-slate-500">No pending high-risk decisions.</div>
            ) : (
              pending.map((c) => (
                <div key={c.id} className="bg-slate-950 border border-slate-800 p-3 rounded flex flex-col gap-2">
                  <div className="flex justify-between items-center">
                    <RiskBadge level={c.riskLevel} />
                    <span className="text-xs font-mono text-slate-500">{c.timestamp}</span>
                  </div>
                  <div>
                    <div className="text-sm font-bold text-slate-200">{c.id} &bull; {c.location}</div>
                    <div className="text-xs text-slate-400 mt-0.5">{c.detection} (confidence {c.confidence}%)</div>
                  </div>
                  <div className="text-xs text-cyan-400">Recommended: {c.recommendation}</div>
                  <div className="flex gap-2 mt-1">
                    <button onClick={() => { setSelectedId(c.id); window.scrollTo({ top: 0, behavior: 'smooth' }); }} className={`text-xs bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded text-slate-300 font-semibold transition-colors ${focusRing}`}>Review feed</button>
                    {c.correlatedCrime && (
                      <button onClick={() => openCase(crimeId(c.correlatedCrime))} className={`text-xs bg-indigo-900/30 border border-indigo-900/50 hover:bg-indigo-900/50 px-3 py-1.5 rounded text-indigo-300 font-semibold transition-colors ${focusRing}`}>Case file</button>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </Panel>

        <div className="grid grid-rows-2 gap-6 h-[380px]">
          <Panel title="AI cross-source correlation" className="!p-4">
            <div className="flex-1 flex flex-col justify-center min-h-0">
              {selectedCamera?.correlatedCrime ? (
                <div className="bg-indigo-950/10 border border-indigo-900/30 p-3 rounded text-center flex flex-col items-center">
                  <span className="text-xs font-bold text-cyan-400">{selectedCamera.id}</span>
                  <span className="text-[11px] text-slate-400">{selectedCamera.detection}</span>
                  <span className="text-slate-600" aria-hidden="true">↓</span>
                  <span className="text-xs font-bold text-red-400">{crimeLabel(selectedCamera.correlatedCrime)}</span>
                  <span className="text-[11px] text-slate-400 font-mono">{crimeId(selectedCamera.correlatedCrime)}</span>
                  <div className="mt-2 text-xs bg-indigo-900/30 text-indigo-300 px-3 py-1 rounded border border-indigo-800/50 font-semibold">
                    Correlation score: {Math.min(99, selectedCamera.confidence + 5)}%
                  </div>
                </div>
              ) : (
                <div className="text-center text-sm text-slate-500">Select a camera with a linked crime to see the correlation.</div>
              )}
            </div>
          </Panel>

          <Panel title="Live AI detection feed" className="!p-4 overflow-hidden">
            <div className="flex-1 overflow-y-auto custom-scrollbar pr-2 space-y-2 min-h-0">
              {recentDetections.length === 0 ? (
                <div className="text-sm text-slate-500">No detections yet.</div>
              ) : recentDetections.map((c) => (
                <button key={c.id} onClick={() => setSelectedId(c.id)} className={`w-full text-left flex justify-between items-center gap-3 bg-slate-950 border p-2 rounded transition-colors ${focusRing} ${selectedCamera?.id === c.id ? 'border-cyan-500/50' : 'border-slate-800 hover:border-slate-600'}`}>
                  <div className="flex items-center gap-3 min-w-0">
                    <span className="text-[11px] font-mono text-slate-500 shrink-0">{c.timestamp}</span>
                    <div className="min-w-0">
                      <div className="text-xs font-bold text-slate-300">{c.id}</div>
                      <div className="text-[11px] text-slate-400 truncate">{c.detection}</div>
                    </div>
                  </div>
                  <RiskBadge level={c.riskLevel} />
                </button>
              ))}
            </div>
          </Panel>
        </div>
      </div>

      {/* TERTIARY ROW */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Panel title="City crime threat assessment" icon={<ShieldIcon />}>
          <div className="space-y-4">
            <div>
              <Label>Current threat</Label>
              <span className={`text-xl font-black uppercase tracking-wider ${cityThreat.color}`}>{cityThreat.level}</span>
            </div>
            <div>
              <Label>AI assessment</Label>
              <p className="text-xs text-slate-300 leading-relaxed">Recent crime activity combined with elevated CCTV detections indicates {cityThreat.level.toLowerCase()} operational risk. Priority sectors require continued monitoring.</p>
            </div>
            <div className="pt-2 border-t border-slate-800">
              <Label>Recommended protocols</Label>
              <ul className="text-xs text-slate-400 list-disc ml-4 space-y-1">
                <li>Increase CCTV monitoring in hotspots</li>
                <li>Verify nearby patrol availability</li>
                <li>Cross-reference active investigations</li>
              </ul>
            </div>
          </div>
        </Panel>

        <Panel title="Camera network health">
          <div className="space-y-5">
            <div className="flex justify-between items-center text-sm font-bold text-slate-200">
              <span>Online node integrity</span>
              <span className="font-mono text-emerald-400">{onlinePct}%</span>
            </div>
            <div className="w-full bg-red-500/80 h-2 rounded overflow-hidden" role="progressbar" aria-valuenow={onlinePct} aria-valuemin={0} aria-valuemax={100} aria-label="Online node integrity">
              <div className="bg-emerald-500 h-full transition-[width] duration-500" style={{ width: `${onlinePct}%` }} />
            </div>
            <div className="grid grid-cols-2 gap-3 text-center">
              <div className="bg-slate-950 border border-slate-800 p-2 rounded">
                <Label>Online</Label>
                <span className="text-sm font-mono font-bold text-emerald-400">{kpis.online}</span>
              </div>
              <div className="bg-slate-950 border border-slate-800 p-2 rounded">
                <Label>Offline</Label>
                <span className="text-sm font-mono font-bold text-red-400">{kpis.monitored - kpis.online}</span>
              </div>
            </div>
          </div>
        </Panel>

        <Panel title="AI agent activity log" className="max-h-[320px]">
          <ol className="flex-1 overflow-y-auto custom-scrollbar pr-2 space-y-4 border-l border-slate-700 ml-1.5 min-h-0">
            {[
              { time: <Clock />, title: 'Continuous vision analysis', body: `Processing ${kpis.online} active streams`, live: true },
              { time: '2 min ago', title: 'Sector risk updated', body: `Threat set to ${cityThreat.level}` },
              { time: '5 min ago', title: 'Database synchronization', body: `Pulled ${kpis.activeCrimes} active records` },
            ].map((e) => (
              <li key={e.title} className="relative pl-5">
                <span className={`absolute -left-[5px] top-1.5 w-2.5 h-2.5 rounded-full border-2 border-slate-900 ${e.live ? 'bg-cyan-500 motion-safe:animate-pulse' : 'bg-slate-600'}`} />
                <div className="text-[11px] font-mono text-slate-500 mb-0.5">{e.time}</div>
                <div className="text-xs font-bold text-slate-300">{e.title}</div>
                <div className="text-xs text-slate-400">{e.body}</div>
              </li>
            ))}
          </ol>
        </Panel>
      </div>

      {isFullscreen && selectedCamera && (
        <FullscreenModal camera={selectedCamera} onClose={closeFullscreen} onOpenCase={openCase} />
      )}
    </div>
  );
}