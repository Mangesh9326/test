import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { calculateModernRoutes } from '../../../services/googleRoutesService';
import { useMapStore } from '../../../store/useMapStore'; // Using existing store for incidents

// Icons
const AlertIcon = () => <svg className="w-5 h-5 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>;
const BrainIcon = () => <svg className="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>;
const MapIcon = () => <svg className="w-5 h-5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" /></svg>;
// ---------- Helpers ----------
const SPEED_COLORS = { NORMAL: '#3b82f6', SLOW: '#f59e0b', TRAFFIC_JAM: '#ef4444' };
const IMPACT = { LIGHT: 'LOW', MODERATE: 'MODERATE', HEAVY: 'HIGH', SEVERE: 'SEVERE' };

// Loads the Maps JS API once (the original only worked if another page had already loaded it)
let mapsLoader = null;
const loadGoogleMaps = (apiKey) => {
  if (window.google?.maps?.importLibrary) return Promise.resolve();
  if (mapsLoader) return mapsLoader;
  mapsLoader = new Promise((resolve, reject) => {
    const fail = (msg) => { mapsLoader = null; reject(new Error(msg)); };
    if (!document.querySelector('script[src*="maps.googleapis.com/maps/api/js"]')) {
      if (!apiKey) return fail('Missing VITE_GOOGLE_MAPS_API_KEY.');
      const s = document.createElement('script');
      s.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&v=weekly&loading=async`;
      s.async = true;
      s.onerror = () => fail('Google Maps script failed to load.');
      document.head.appendChild(s);
    }
    const start = Date.now();
    const t = setInterval(() => {
      if (window.google?.maps?.importLibrary) { clearInterval(t); resolve(); }
      else if (Date.now() - start > 15000) { clearInterval(t); fail('Google Maps timed out while loading.'); }
    }, 100);
  });
  return mapsLoader;
};

// Own polyline decoder: no dependency on the "geometry" library
const decodePolyline = (str = '') => {
  const pts = [];
  let i = 0, lat = 0, lng = 0;
  while (i < str.length) {
    for (let axis = 0; axis < 2; axis++) {
      let b, shift = 0, result = 0;
      do { b = str.charCodeAt(i++) - 63; result |= (b & 0x1f) << shift; shift += 5; } while (b >= 0x20);
      const d = result & 1 ? ~(result >> 1) : result >> 1;
      if (axis === 0) lat += d; else lng += d;
    }
    pts.push({ lat: lat / 1e5, lng: lng / 1e5 });
  }
  return pts;
};

const distM = (a, b) => {
  const R = 6371000, r = Math.PI / 180;
  const dLa = (b.lat - a.lat) * r, dLo = (b.lng - a.lng) * r;
  const h = Math.sin(dLa / 2) ** 2 + Math.cos(a.lat * r) * Math.cos(b.lat * r) * Math.sin(dLo / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
};

const incidentCoords = (inc) => {
  const lat = Number(inc.lat ?? inc.latitude ?? inc.current_lat ?? inc.location?.lat);
  const lng = Number(inc.lng ?? inc.longitude ?? inc.current_lng ?? inc.location?.lng);
  return Number.isFinite(lat) && Number.isFinite(lng) ? { lat, lng } : null;
};

const isActiveIncident = (inc) =>
  !['resolved', 'closed', 'cleared'].includes(String(inc.status || '').toLowerCase());

const congestion = (dur, stat) => {
  const r = stat > 0 ? dur / stat : 1;
  return r < 1.15 ? 'LIGHT' : r < 1.4 ? 'MODERATE' : r < 1.8 ? 'HEAVY' : 'SEVERE';
};
const fmtMin = (s) => `${Math.max(1, Math.round(s / 60))} mins`;

// Built-in analysis (used when the backend Traffic Agent is unreachable, and as a base for its response)
const localEvaluate = (routes, active) => {
  const scored = routes.map((r, idx) => {
    const path = decodePolyline(r.polyline);
    const hits = active
      .filter((inc) => { const c = incidentCoords(inc); return c && path.some((p) => distM(p, c) < 300); })
      .map((inc) => ({
        type: inc.type || inc.incident_type || 'Traffic Incident',
        severity: String(inc.severity || 'HIGH').toUpperCase(),
        location: inc.location?.name || (typeof inc.location === 'string' ? inc.location : null) || inc.address || inc.name || 'Along corridor',
        impact: inc.impact || 'Delays likely',
      }));
    const penalty = hits.reduce((s, h) => s + (h.severity === 'CRITICAL' ? 300 : 150), 0);
    return { idx, hits, score: r.durationSeconds + penalty };
  });
  const best = scored.reduce((a, b) => (b.score < a.score ? b : a));
  const r = routes[best.idx];
  const level = congestion(r.durationSeconds, r.staticDurationSeconds);
  const savedMin = Math.round((scored[0].score - best.score) / 60);
  return {
    incidentsByRoute: scored.map((s) => s.hits),
    decision: {
      selectedRouteIndex: best.idx,
      impact: IMPACT[level],
      recommendation: `${r.name} · ${r.summary} · ${r.duration}`,
      rationale:
        `${r.name} has the lowest effective travel time (${r.duration}` +
        `${r.delayMinutes > 0 ? `, including ${r.delayMinutes} min of traffic delay` : ''}). ` +
        (best.hits.length ? `${best.hits.length} active incident(s) are near this corridor.` : 'No active incidents were found near this corridor.'),
      outcome: best.idx !== 0 && savedMin > 0
        ? `Avoids roughly ${savedMin} min compared with the default route.`
        : 'Default route is already the best option given current traffic and incidents.',
    },
  };
};

// Real Google predictive durations for the recommended route (+15 / +30 min departures)
const buildPrediction = async (route, o, d, apiKey) => {
  const at = (m) => new Date(Date.now() + m * 60000).toISOString();
  const [a, b] = await Promise.all([15, 30].map((m) => calculateModernRoutes(o, d, apiKey, { departureTime: at(m) })));
  const pick = (list) => list.reduce((best, r) =>
    !best || Math.abs(r.rawRoute.distanceMeters - route.rawRoute.distanceMeters) < Math.abs(best.rawRoute.distanceMeters - route.rawRoute.distanceMeters) ? r : best, null);
  const fmt = (r) => `${congestion(r.durationSeconds, r.staticDurationSeconds)} · ${r.duration}`;
  const pa = pick(a), pb = pick(b);
  if (!pa || !pb) throw new Error('No predictive routes');
  return {
    current: fmt(route),
    plus15: fmt(pa),
    plus30: fmt(pb),
    confidence: 'Google predictive traffic (recommended route)',
  };
};

// Same thresholds as the backend's googleTrafficLevel()
const backendTrafficLevel = (delayMin, staticSec) => {
  const staticMin = staticSec / 60;
  if (!staticMin) return 'LOW';
  const r = delayMin / staticMin;
  return r >= 0.6 ? 'SEVERE' : r >= 0.35 ? 'HIGH' : r >= 0.15 ? 'MODERATE' : 'LOW';
};

// Routes in the same shape the backend's computeGoogleRoute() produces
const toBackendRoute = ({ rawRoute, distance, duration, ...r }) => ({
  id: r.id,
  name: r.name,
  summary: r.summary,
  distanceKm: Number.parseFloat(distance) || 0,
  durationSeconds: r.durationSeconds,
  staticDurationSeconds: r.staticDurationSeconds,
  delayMinutes: r.delayMinutes,
  averageSpeedKmh: r.averageSpeedKmh,
  trafficLevel: backendTrafficLevel(r.delayMinutes, r.staticDurationSeconds),
  polyline: r.polyline,
  speedReadingIntervals: r.speedReadingIntervals,
  source: 'GOOGLE_ROUTES_API',
  isPredicted: r.isPredicted,
});

// Backend returns { status, decision: { action, severity, metadata }, record }.
// Map only the fields that are actually present; anything missing falls back to the built-in analysis.
const adaptAgentResponse = (raw) => {
  const d = raw?.decision;
  if (!d) return null;
  const p = d.metadata?.payload || d.metadata || {};
  const decision = {};
  if (Number.isInteger(p.selectedRouteIndex)) decision.selectedRouteIndex = p.selectedRouteIndex;
  if (d.action) decision.recommendation = d.action;
  const why = d.metadata?.reasoning || p.rationale;
  if (why) decision.rationale = why;
  if (p.outcome) decision.outcome = p.outcome;
  if (p.impact) decision.impact = String(p.impact).toUpperCase();
  if (p.confidence) decision.confidence = String(p.confidence);
  return {
    decision,
    incidents: Array.isArray(p.incidents) ? p.incidents : null,
    prediction: p.prediction?.plus15 ? p.prediction : null,
    signal: p.signal && typeof p.signal === 'object' ? p.signal : null,
  };
};

export default function RouteIntelligence() {
  const mapRef = useRef(null);
  const polylinesRef = useRef([]);
  const requestRef = useRef(0);
  const [mapObj, setMapObj] = useState(null);
  const [mapError, setMapError] = useState(null);

  const incidents = useMapStore((state) => state.incidents);

  const [origin, setOrigin] = useState("");
  const [destination, setDestination] = useState("");

  // READY, CALCULATING, EVALUATING, COMPLETE, ERROR
  const [uiState, setUiState] = useState("READY");
  const [errorMessage, setErrorMessage] = useState(null);
  const [notice, setNotice] = useState(null);

  const [routes, setRoutes] = useState([]);
  const [selectedRouteIndex, setSelectedRouteIndex] = useState(0);

  const [agentDecision, setAgentDecision] = useState(null);
  const [analysisSource, setAnalysisSource] = useState('agent');
  const [localIncidents, setLocalIncidents] = useState([]);
  const [backendIncidents, setBackendIncidents] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [signalData, setSignalData] = useState(null);

  const routeIncidents = backendIncidents?.length ? backendIncidents : (localIncidents[selectedRouteIndex] || []);
  const paths = useMemo(() => routes.map((r) => decodePolyline(r.polyline)), [routes]);

  // Initialize map (loads the Maps API itself if needed)
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await loadGoogleMaps(import.meta.env.VITE_GOOGLE_MAPS_API_KEY);
        const { Map, TrafficLayer } = await window.google.maps.importLibrary("maps");
        if (cancelled || !mapRef.current) return;
        const m = new Map(mapRef.current, {
          center: { lat: 19.0760, lng: 72.8777 },
          zoom: 12,
          backgroundColor: '#0f172a',
          styles: [
            { elementType: "geometry", stylers: [{ color: "#1e293b" }] },
            { elementType: "labels.text.stroke", stylers: [{ color: "#0f172a" }] },
            { elementType: "labels.text.fill", stylers: [{ color: "#94a3b8" }] },
            { featureType: "road", elementType: "geometry", stylers: [{ color: "#334155" }] },
            { featureType: "road.highway", elementType: "geometry", stylers: [{ color: "#475569" }] }
          ]
        });
        new TrafficLayer().setMap(m);
        setMapObj(m);
      } catch (err) {
        console.error("Map initialization failed", err);
        if (!cancelled) setMapError(err.message || "Unable to load map or traffic layer.");
      }
    })();
    return () => {
      cancelled = true;
      polylinesRef.current.forEach((p) => p.setMap(null));
      polylinesRef.current = [];
    };
  }, []);

  const clearRoutes = useCallback(() => {
    polylinesRef.current.forEach((p) => p.setMap(null));
    polylinesRef.current = [];
  }, []);

  // Draw routes; the selected one is coloured by Google's speed intervals (blue / amber / red)
  useEffect(() => {
    if (!mapObj) return;
    clearRoutes();
    routes.forEach((route, idx) => {
      const path = paths[idx];
      if (!path?.length) return;
      const sel = idx === selectedRouteIndex;
      const add = (p, color) => {
        const pl = new window.google.maps.Polyline({
          path: p, strokeColor: color, strokeWeight: sel ? 6 : 4,
          strokeOpacity: sel ? 1 : 0.5, zIndex: sel ? 10 : 1, map: mapObj,
        });
        pl.addListener('click', () => setSelectedRouteIndex(idx));
        polylinesRef.current.push(pl);
      };
      if (sel && route.speedReadingIntervals?.length) {
        route.speedReadingIntervals.forEach((iv) => {
          const seg = path.slice(iv.startPolylinePointIndex || 0, (iv.endPolylinePointIndex ?? path.length - 1) + 1);
          if (seg.length > 1) add(seg, SPEED_COLORS[iv.speed] || SPEED_COLORS.NORMAL);
        });
      } else {
        add(path, sel ? SPEED_COLORS.NORMAL : '#64748b');
      }
    });
  }, [mapObj, routes, paths, selectedRouteIndex, clearRoutes]);

  // Fit the viewport only when new routes arrive (not on every selection change)
  useEffect(() => {
    if (!mapObj || !paths.length) return;
    const bounds = new window.google.maps.LatLngBounds();
    paths.forEach((p) => p.forEach((pt) => bounds.extend(pt)));
    if (!bounds.isEmpty()) mapObj.fitBounds(bounds, 60);
  }, [mapObj, paths]);

  const handleRouteRequest = async (e) => {
    if (e) e.preventDefault();
    if (!origin.trim() || !destination.trim()) {
      setErrorMessage("Both origin and destination are required.");
      setUiState("ERROR");
      return;
    }

    const reqId = ++requestRef.current;
    const o = origin.trim(), d = destination.trim();
    setUiState("CALCULATING");
    setErrorMessage(null);
    setNotice(null);
    setRoutes([]);
    setSelectedRouteIndex(0);
    setAgentDecision(null);
    setBackendIncidents(null);
    setLocalIncidents([]);
    setPredictionData(null);
    setSignalData(null);

    const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
    let calculated;
    try {
      calculated = await calculateModernRoutes(o, d, apiKey);
      if (reqId !== requestRef.current) return;
      if (!calculated.length) {
        setErrorMessage("No valid routes found for these locations.");
        setUiState("ERROR");
        return;
      }
    } catch (error) {
      if (reqId !== requestRef.current) return;
      setErrorMessage(error.message || "Route calculation failed.");
      setUiState("ERROR");
      return;
    }

    // Routes are usable immediately; intelligence layers follow
    setRoutes(calculated);
    setUiState("EVALUATING");

    const active = (incidents || []).filter(isActiveIncident);
    const local = localEvaluate(calculated, active);

    let backend = null;
    try {
      const res = await fetch("/api/traffic/routes/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          origin: o,
          destination: d,
          routes: calculated.map(toBackendRoute),
          source: "GOOGLE_ROUTES_API",
          activeIncidents: active,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      backend = adaptAgentResponse(await res.json());
    } catch (err) {
      console.warn("Traffic Agent unavailable, using built-in analysis:", err);
    }
    if (reqId !== requestRef.current) return;

    const decision = { ...local.decision, ...(backend?.decision || {}) };
    const chosen = Math.min(Math.max(Number(decision.selectedRouteIndex) || 0, 0), calculated.length - 1);
    decision.selectedRouteIndex = chosen;

    let prediction = backend?.prediction || null;
    if (!prediction) {
      try {
        prediction = await buildPrediction(calculated[chosen], o, d, apiKey);
      } catch (err) {
        console.warn("Predictive traffic unavailable:", err);
      }
      if (reqId !== requestRef.current) return;
    }

    setAnalysisSource(backend ? 'agent' : 'local');
    if (!backend) setNotice("Traffic Agent service is unreachable. Showing built-in analysis from Google traffic data and live incidents.");
    setAgentDecision(decision);
    setSelectedRouteIndex(chosen);
    setLocalIncidents(local.incidentsByRoute);
    setBackendIncidents(Array.isArray(backend?.incidents) ? backend.incidents : null);
    setPredictionData(prediction);
    setSignalData(backend?.signal || null);
    setUiState("COMPLETE");
  };

  // Reads the normalized shape produced by googleRoutesService
  const getRouteMetrics = (route) => ({
    distance: route.distance || "Unknown Distance",
    duration: route.duration || "Unknown Duration",
  });

  return (
    <div className="flex flex-col gap-6 text-slate-100 p-6 h-full overflow-y-auto custom-scrollbar font-sans bg-slate-950">
      
      {/* HEADER */}
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4 shrink-0 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-2xl font-bold uppercase tracking-tight text-white flex items-center gap-2">
            Route Intelligence
          </h1>
          <p className="text-sm text-slate-400 mt-1">AI-powered route planning, traffic prediction and incident-aware routing</p>
        </div>
        <div className="flex items-center gap-3">
          <span className={`px-3 py-1.5 border text-xs font-bold uppercase rounded flex items-center gap-2 ${uiState === 'EVALUATING' ? 'bg-amber-900/30 border-amber-500 text-amber-400' : 'bg-blue-900/30 border-blue-500 text-blue-400'}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${uiState === 'EVALUATING' ? 'bg-amber-400 animate-pulse' : 'bg-blue-400'}`}></span>
            {uiState === 'EVALUATING' ? 'TRAFFIC AGENT EVALUATING...' : analysisSource === 'local' ? 'LOCAL ANALYSIS' : 'TRAFFIC AGENT ACTIVE'}
          </span>
        </div>
      </div>

      {/* INPUT PANEL */}
      <div className="bg-slate-900 p-5 rounded-xl border border-slate-800 shrink-0 shadow-lg">
        <form onSubmit={handleRouteRequest} className="flex flex-col md:flex-row items-center gap-4">
          <div className="flex-1 w-full">
            <label className="text-[10px] text-slate-500 uppercase font-bold tracking-widest block mb-1">Origin</label>
            <input 
              value={origin} 
              onChange={(e) => setOrigin(e.target.value)} 
              placeholder="e.g., Dadar Station, Mumbai"
              className="bg-slate-950 border border-slate-700 p-2.5 rounded w-full text-sm outline-none focus:border-blue-500 transition-colors" 
              required
            />
          </div>
          <div className="hidden md:block text-slate-600 mt-4">→</div>
          <div className="flex-1 w-full">
            <label className="text-[10px] text-slate-500 uppercase font-bold tracking-widest block mb-1">Destination</label>
            <input 
              value={destination} 
              onChange={(e) => setDestination(e.target.value)} 
              placeholder="e.g., Wadala, Mumbai"
              className="bg-slate-950 border border-slate-700 p-2.5 rounded w-full text-sm outline-none focus:border-blue-500 transition-colors" 
              required
            />
          </div>
          <div className="w-full md:w-auto mt-4 md:mt-5">
            <button 
              type="submit"
              disabled={uiState === "CALCULATING" || uiState === "EVALUATING"}
              className="bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-500 text-white w-full md:w-auto px-6 py-2.5 rounded text-sm font-bold uppercase tracking-wider transition-colors"
            >
              {uiState === "CALCULATING" ? "Calculating..." : "Calculate Route"}
            </button>
          </div>
        </form>
      </div>

      {notice && (
        <div className="bg-amber-900/20 border border-amber-500/60 text-amber-300 p-3 rounded-xl text-xs font-medium shrink-0">{notice}</div>
      )}

      {/* ERROR MESSAGE */}
      {errorMessage && (
        <div className="bg-red-900/20 border border-red-500 text-red-400 p-4 rounded-xl flex items-center gap-3 shrink-0">
          <AlertIcon />
          <span className="text-sm font-medium">{errorMessage}</span>
        </div>
      )}

      {/* OPERATIONAL WORKFLOW STATES */}
      {uiState !== "READY" && uiState !== "ERROR" && (
        <div className="flex justify-between items-center text-[10px] font-bold uppercase tracking-widest text-slate-500 bg-slate-900/50 p-3 rounded-lg border border-slate-800 shrink-0">
          <div className={`flex items-center gap-2 ${uiState === 'CALCULATING' ? 'text-blue-400 animate-pulse' : 'text-slate-300'}`}>
            <span>1. Route Calculation</span>
          </div>
          <span>→</span>
          <div className={`flex items-center gap-2 ${uiState === 'EVALUATING' ? 'text-blue-400 animate-pulse' : (uiState === 'COMPLETE' ? 'text-slate-300' : 'text-slate-600')}`}>
            <span>2. Inspect Traffic Impact</span>
          </div>
          <span>→</span>
          <div className={`flex items-center gap-2 ${uiState === 'COMPLETE' ? 'text-emerald-400' : 'text-slate-600'}`}>
            <span>3. Traffic Agent Evaluation Complete</span>
          </div>
        </div>
      )}

      {/* KPI SECTION (Only visible when complete) */}
      {uiState === "COMPLETE" && routes.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 shrink-0">
          <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-center">
            <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Routes Found</span>
            <span className="text-xl font-mono text-slate-100 font-bold mt-1">{routes.length}</span>
          </div>
          <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-center">
            <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Traffic Impact</span>
            <span className={`text-lg font-bold uppercase mt-1 ${agentDecision?.impact === 'HIGH' || agentDecision?.impact === 'SEVERE' ? 'text-red-400' : agentDecision?.impact === 'MODERATE' ? 'text-amber-400' : 'text-emerald-400'}`}>
              {agentDecision?.impact || "UNKNOWN"}
            </span>
          </div>
          <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-center md:col-span-2">
            <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Agent Recommendation</span>
            <span className="text-sm font-bold text-blue-400 mt-1 truncate" title={agentDecision?.recommendation || "N/A"}>
              {agentDecision?.recommendation || "Operator judgment required"}
            </span>
          </div>
          <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex flex-col justify-center">
            <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Active Incidents</span>
            <span className="text-xl font-mono font-bold mt-1 text-red-400">{routeIncidents.length}</span>
          </div>
        </div>
      )}

      {/* MAIN VIEWPORT: Map + Route Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1 min-h-[500px]">
        
        {/* Google Traffic Map */}
        <div className="lg:col-span-7 xl:col-span-8 bg-slate-900 border border-slate-800 rounded-xl shadow-lg relative flex flex-col min-h-[400px] overflow-hidden">
          {mapError ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-red-400 p-6 text-center z-10 bg-slate-950">
              <AlertIcon />
              <span className="mt-2 text-sm font-mono">{mapError}</span>
            </div>
          ) : !mapObj ? (
            <div className="absolute inset-0 flex items-center justify-center text-slate-400 z-10 bg-slate-950">
              <span className="text-sm font-mono animate-pulse">Initializing Map...</span>
            </div>
          ) : null}
          <div ref={mapRef} className="absolute inset-0 w-full h-full" />
          <div className="absolute bottom-1 left-1 z-10 bg-slate-950/90 backdrop-blur border border-slate-700 px-3 py-2 rounded shadow flex items-center gap-2">
            <MapIcon />
            <span className="text-[10px] font-bold text-slate-300 uppercase tracking-widest">Traffic Map</span>
          </div>
        </div>

        {/* Route Comparison */}
        <div className="lg:col-span-5 xl:col-span-4 flex flex-col gap-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col h-full overflow-hidden">
            <h3 className="font-bold text-sm uppercase tracking-widest text-slate-200 border-b border-slate-800 pb-3 mb-3 shrink-0">
              Route Comparison
            </h3>
            
            <div className="flex flex-col gap-3 overflow-y-auto custom-scrollbar flex-1 pr-1">
              {routes.length === 0 ? (
                <div className="text-center text-slate-500 text-sm italic my-auto">
                  {uiState === 'READY' ? 'Calculate a route to view comparisons.' : 'No routes to display.'}
                </div>
              ) : (
                routes.map((route, idx) => {
                  const isSelected = idx === selectedRouteIndex;
                  const isAgentRecommended = agentDecision && agentDecision.selectedRouteIndex === idx;
                  const metrics = getRouteMetrics(route);

                  return (
                    <div 
                      key={idx} 
                      onClick={() => setSelectedRouteIndex(idx)}
                      className={`p-4 rounded-lg border cursor-pointer transition-all ${isSelected ? 'bg-blue-900/20 border-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.1)]' : 'bg-slate-950 border-slate-800 hover:border-slate-700'}`}
                    >
                      <div className="flex justify-between items-start mb-2">
                        <div className="font-bold text-slate-100 text-sm">{route.summary || `Route Option ${idx + 1}`}</div>
                        {isAgentRecommended && (
                          <span className="text-[9px] bg-blue-500/10 text-blue-400 px-2 py-0.5 rounded uppercase tracking-widest border border-blue-500/20">Recommended</span>
                        )}
                      </div>
                      
                      <div className="grid grid-cols-2 gap-2 mb-2 text-xs">
                        <div>
                          <span className="text-slate-500 uppercase tracking-widest text-[9px] block">Distance</span>
                          <span className="font-mono text-slate-300">{metrics.distance}</span>
                        </div>
                        <div>
                          <span className="text-slate-500 uppercase tracking-widest text-[9px] block">Duration</span>
                          <span className={`font-mono font-bold ${isSelected ? 'text-blue-400' : 'text-slate-300'}`}>{metrics.duration}</span>
                        </div>
                      </div>

                      <div className="text-[10px] font-bold tracking-widest uppercase mt-3 pt-2 border-t border-slate-800/50 flex justify-between items-center">
                         <span className="text-slate-500">Status</span>
                         <span className={isSelected ? 'text-blue-400' : 'text-slate-400'}>{isSelected ? 'SELECTED' : 'ALTERNATE'}</span>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>

      </div>

      {/* INTELLIGENCE PANELS */}
      {uiState === "COMPLETE" && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6 shrink-0 mt-2">
          
          {/* Traffic Agent Decision */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col">
            <h3 className="font-bold text-sm uppercase tracking-widest text-blue-400 flex items-center gap-2 mb-4">
              <BrainIcon /> Traffic Agent
            </h3>
            {agentDecision ? (
              <div className="flex flex-col gap-3 flex-1 text-sm">
                <div>
                  <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest block mb-1">Decision Rationale</span>
                  <p className="text-slate-300">{agentDecision.rationale || "Evaluated multiple routes and selected the optimal path based on current flow constraints."}</p>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest block mb-1">Expected Outcome</span>
                  <p className="text-emerald-400 font-medium">{agentDecision.outcome || "Optimized travel time and minimized network impact."}</p>
                </div>
                <div className="mt-auto pt-3 border-t border-slate-800 flex justify-between items-center text-[10px] font-bold uppercase tracking-widest">
                  <span className="text-slate-500">Confidence</span>
                  <span className="text-slate-200">{agentDecision.confidence || "N/A"}</span>
                </div>
              </div>
            ) : (
              <div className="text-slate-500 italic text-sm m-auto">Evaluation unavailable.</div>
            )}
          </div>

          {/* Active Incidents */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col">
            <h3 className="font-bold text-sm uppercase tracking-widest text-amber-500 flex items-center gap-2 mb-4">
              <AlertIcon /> Route Impact
            </h3>
            {routeIncidents.length > 0 ? (
              <div className="flex flex-col gap-3 flex-1 overflow-y-auto custom-scrollbar pr-1">
                {routeIncidents.map((inc, i) => (
                  <div key={i} className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <div className="flex justify-between items-start mb-1">
                      <span className="text-xs font-bold text-slate-200 uppercase">{inc.type || "Traffic Incident"}</span>
                      <span className={`text-[9px] px-1.5 py-0.5 rounded uppercase tracking-widest ${inc.severity === 'CRITICAL' ? 'bg-red-500/20 text-red-400' : 'bg-amber-500/20 text-amber-400'}`}>{inc.severity || "HIGH"}</span>
                    </div>
                    <div className="text-xs text-slate-400 truncate">{inc.location || "Along corridor"}</div>
                    <div className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mt-2 border-t border-slate-800 pt-1">
                      Impact: <span className="text-amber-400 ml-1">{inc.impact || "Delays likely"}</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-slate-500 italic text-sm m-auto text-center px-4">No active traffic incidents affecting this route.</div>
            )}
          </div>

          {/* Traffic Prediction */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col">
            <h3 className="font-bold text-sm uppercase tracking-widest text-indigo-400 mb-4">Traffic Prediction</h3>
            {predictionData ? (
              <div className="flex flex-col gap-3 flex-1">
                <div className="flex justify-between items-center bg-slate-950 p-3 rounded border border-slate-800">
                  <span className="text-xs text-slate-400 font-bold uppercase tracking-widest">Current</span>
                  <span className="text-sm font-bold text-slate-200 uppercase">{predictionData.current || "Unknown"}</span>
                </div>
                <div className="flex justify-between items-center bg-slate-950 p-3 rounded border border-slate-800">
                  <span className="text-xs text-slate-400 font-bold uppercase tracking-widest">+15 Min</span>
                  <span className="text-sm font-bold text-slate-200 uppercase">{predictionData.plus15 || "Unknown"}</span>
                </div>
                <div className="flex justify-between items-center bg-slate-950 p-3 rounded border border-slate-800">
                  <span className="text-xs text-slate-400 font-bold uppercase tracking-widest">+30 Min</span>
                  <span className="text-sm font-bold text-slate-200 uppercase">{predictionData.plus30 || "Unknown"}</span>
                </div>
                <div className="mt-auto pt-3 border-t border-slate-800 flex justify-between items-center text-[10px] font-bold uppercase tracking-widest">
                  <span className="text-slate-500">Model Confidence</span>
                  <span className="text-slate-200">{predictionData.confidence || "N/A"}</span>
                </div>
              </div>
            ) : (
              <div className="text-slate-500 italic text-sm m-auto text-center">Traffic prediction unavailable for this corridor.</div>
            )}
          </div>

          {/* Signal Intelligence */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col">
            <h3 className="font-bold text-sm uppercase tracking-widest text-emerald-400 mb-4">Signal Intelligence</h3>
            {signalData ? (
              <div className="flex flex-col gap-3 flex-1 text-sm">
                <div>
                  <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest block mb-1">Affected Intersections</span>
                  <span className="text-slate-200 font-medium">{signalData.intersections?.join(", ") || "Unknown corridor nodes"}</span>
                </div>
                <div className="bg-slate-950 p-3 rounded border border-slate-800 mt-2">
                  <div className="flex justify-between text-xs mb-2">
                    <span className="text-slate-400 font-bold uppercase">Current Timing</span>
                    <span className="font-mono text-slate-300">G: {signalData.currentGreen || '--'}s | R: {signalData.currentRed || '--'}s</span>
                  </div>
                  <div className="flex justify-between text-xs border-t border-slate-800 pt-2">
                    <span className="text-blue-400 font-bold uppercase">Agent Rec</span>
                    <span className="font-mono font-bold text-emerald-400">G: {signalData.recGreen || '--'}s | R: {signalData.recRed || '--'}s</span>
                  </div>
                </div>
                <div className="mt-auto pt-3 border-t border-slate-800 flex justify-between items-center text-[10px] font-bold uppercase tracking-widest">
                  <span className="text-slate-500">Status</span>
                  <span className="text-blue-400">{signalData.status || "MONITORING"}</span>
                </div>
              </div>
            ) : (
              <div className="text-slate-500 italic text-sm m-auto text-center px-4">No signal-control data available for this route.</div>
            )}
          </div>

        </div>
      )}
    </div>
  );
}