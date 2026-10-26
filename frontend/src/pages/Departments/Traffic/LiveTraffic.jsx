import { useEffect, useRef, useState, useMemo } from "react";
import React from "react";
import { useMapStore } from "../../../store/useMapStore";
import { calculateModernRoutes } from "../../../services/googleRoutesService";

// UI Icons
const MapPinIcon = () => (
  <svg
    className="w-5 h-5 text-red-400"
    fill="none"
    viewBox="0 0 24 24"
    stroke="currentColor"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
    />
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
    />
  </svg>
);
const BrainIcon = () => (
  <svg
    className="w-5 h-5 text-blue-400"
    fill="none"
    viewBox="0 0 24 24"
    stroke="currentColor"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
    />
  </svg>
);
const SignalIcon = () => (
  <svg
    className="w-5 h-5 text-emerald-400"
    fill="none"
    viewBox="0 0 24 24"
    stroke="currentColor"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2zM12 14h.01M12 10h.01"
    />
  </svg>
);
const CheckIcon = () => (
  <svg
    className="w-4 h-4 text-emerald-400"
    fill="none"
    viewBox="0 0 24 24"
    stroke="currentColor"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M5 13l4 4L19 7"
    />
  </svg>
);
const AlertIcon = () => (
  <svg
    className="w-4 h-4 text-red-400"
    fill="none"
    viewBox="0 0 24 24"
    stroke="currentColor"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
    />
  </svg>
);
const TerminalIcon = () => (
  <svg
    className="w-4 h-4 text-green-400"
    fill="none"
    viewBox="0 0 24 24"
    stroke="currentColor"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
    />
  </svg>
);

const EMPTY_METRICS = {
  signalOptimizations: 0,
  successfulRecommendations: 0,
  routeRecommendations: 0,
  activeInterventions: 0,
  predictionAlerts: 0,
};
// Kept for internal step tracking (used to drive the terminal log only — no longer rendered as a timeline)
const SCENARIO_STEPS = [
  "DETECT",
  "ANALYZE",
  "PREDICT",
  "DECIDE",
  "OPTIMIZE",
  "ROUTE",
  "OUTCOME",
];

// ---------------------------------------------------------------------
// MUMBAI DEFAULT / DEMO DATASET
// Shown whenever there is no active scenario (and as a fallback if the
// backend's live telemetry endpoint is unreachable) so the dashboard never
// looks empty. This is hardcoded, clearly-labeled demo data — never
// presented as genuine real-time telemetry. Coordinates correspond to the
// actual named areas/junctions.
// ---------------------------------------------------------------------
const MUMBAI_CENTER = { lat: 19.076, lng: 72.8777 };
const MUMBAI_DEFAULT_ZOOM = 12;

const SEVERITY_LADDER = ["LOW", "MODERATE", "HIGH", "SEVERE"];
const SEVERITY_RANK = { CRITICAL: 0, SEVERE: 0, HIGH: 1, MODERATE: 2, LOW: 3 };

function severityTextColor(severity) {
  return severity === "CRITICAL" || severity === "SEVERE"
    ? "text-red-500"
    : severity === "HIGH"
      ? "text-orange-400"
      : severity === "MODERATE"
        ? "text-yellow-400"
        : "text-emerald-400";
}

function severityMarkerHex(severity) {
  return severity === "CRITICAL" || severity === "SEVERE"
    ? "#ef4444"
    : severity === "HIGH"
      ? "#fb923c"
      : severity === "MODERATE"
        ? "#facc15"
        : "#34d399";
}

function nextSeverity(severity) {
  const idx = SEVERITY_LADDER.indexOf(
    severity === "CRITICAL" ? "SEVERE" : severity,
  );
  const safeIdx = idx === -1 ? 1 : idx;
  return SEVERITY_LADDER[Math.min(SEVERITY_LADDER.length - 1, safeIdx + 1)];
}

// Believable default signal timing, varied by congestion severity — never
// all-identical, never obviously fake (no 0s/999s).
function defaultSignalTiming(severity) {
  switch (severity) {
    case "CRITICAL":
    case "SEVERE":
      return { currentGreen: 28, currentRed: 52, optGreen: 42, optRed: 38 };
    case "HIGH":
      return { currentGreen: 32, currentRed: 48, optGreen: 40, optRed: 40 };
    case "MODERATE":
      return { currentGreen: 34, currentRed: 42, optGreen: 37, optRed: 39 };
    default:
      return { currentGreen: 36, currentRed: 36, optGreen: 36, optRed: 36 };
  }
}

const RAW_DEFAULT_MUMBAI_LOCATIONS = [
  { id: "default-dadar", name: "Dadar", lat: 19.0178, lng: 72.8478, severity: "HIGH", averageSpeed: 19, delayMinutes: 8, vehicleCount: 1420, trend: "Increasing" },
  { id: "default-bandra", name: "Bandra", lat: 19.0596, lng: 72.8295, severity: "MODERATE", averageSpeed: 26, delayMinutes: 5, vehicleCount: 1080, trend: "Stable" },
  { id: "default-bkc", name: "Bandra Kurla Complex", lat: 19.0665, lng: 72.8686, severity: "MODERATE", averageSpeed: 31, delayMinutes: 4, vehicleCount: 960, trend: "Stable" },
  { id: "default-sion", name: "Sion", lat: 19.043, lng: 72.8619, severity: "SEVERE", averageSpeed: 14, delayMinutes: 12, vehicleCount: 1610, trend: "Increasing" },
  { id: "default-wadala", name: "Wadala", lat: 19.0176, lng: 72.8562, severity: "MODERATE", averageSpeed: 24, delayMinutes: 6, vehicleCount: 890, trend: "Stable" },
  { id: "default-kurla", name: "Kurla", lat: 19.0728, lng: 72.8826, severity: "HIGH", averageSpeed: 18, delayMinutes: 9, vehicleCount: 1340, trend: "Increasing" },
  { id: "default-andheri", name: "Andheri", lat: 19.1197, lng: 72.8468, severity: "HIGH", averageSpeed: 20, delayMinutes: 8, vehicleCount: 1290, trend: "Stable" },
  { id: "default-powai", name: "Powai", lat: 19.1176, lng: 72.906, severity: "LOW", averageSpeed: 34, delayMinutes: 2, vehicleCount: 540, trend: "Decreasing" },
  { id: "default-lower-parel", name: "Lower Parel", lat: 18.996, lng: 72.8302, severity: "MODERATE", averageSpeed: 25, delayMinutes: 5, vehicleCount: 870, trend: "Stable" },
  { id: "default-worli", name: "Worli", lat: 19.0096, lng: 72.8175, severity: "MODERATE", averageSpeed: 27, delayMinutes: 4, vehicleCount: 760, trend: "Stable" },
  { id: "default-chembur", name: "Chembur", lat: 19.0522, lng: 72.9006, severity: "LOW", averageSpeed: 32, delayMinutes: 3, vehicleCount: 610, trend: "Stable" },
  { id: "default-vikhroli", name: "Vikhroli", lat: 19.109, lng: 72.925, severity: "LOW", averageSpeed: 33, delayMinutes: 2, vehicleCount: 480, trend: "Decreasing" },
  { id: "default-ghatkopar", name: "Ghatkopar", lat: 19.0864, lng: 72.9081, severity: "HIGH", averageSpeed: 21, delayMinutes: 7, vehicleCount: 1150, trend: "Increasing" },
  { id: "default-santacruz", name: "Santacruz", lat: 19.0821, lng: 72.8416, severity: "MODERATE", averageSpeed: 28, delayMinutes: 4, vehicleCount: 720, trend: "Stable" },
  { id: "default-airport", name: "Mumbai Airport (CSMIA)", lat: 19.0896, lng: 72.8656, severity: "HIGH", averageSpeed: 22, delayMinutes: 10, vehicleCount: 1080, trend: "Increasing" },
  { id: "default-fort", name: "Fort", lat: 18.9345, lng: 72.8358, severity: "LOW", averageSpeed: 30, delayMinutes: 2, vehicleCount: 430, trend: "Stable" },
  { id: "default-nariman-point", name: "Nariman Point", lat: 18.9256, lng: 72.8242, severity: "LOW", averageSpeed: 29, delayMinutes: 2, vehicleCount: 390, trend: "Stable" },
  { id: "default-marine-drive", name: "Marine Drive", lat: 18.9432, lng: 72.8234, severity: "LOW", averageSpeed: 35, delayMinutes: 1, vehicleCount: 350, trend: "Decreasing" },
];

const DEFAULT_MUMBAI_LOCATIONS = RAW_DEFAULT_MUMBAI_LOCATIONS.map((loc) => ({
  ...loc,
  color: severityTextColor(loc.severity),
  trafficLevel: loc.severity,
  source: "DEMO TELEMETRY",
  signal: {
    signalId: loc.id,
    currentGreenTime: defaultSignalTiming(loc.severity).currentGreen,
    currentRedTime: defaultSignalTiming(loc.severity).currentRed,
  },
}));

// ---------------------------------------------------------------------
// PEAK-TIME TRAFFIC FORECAST
// Google Maps has no public API for genuine *past* traffic (the live
// TrafficLayer is current-only, and the Routes API's traffic-aware duration
// only accepts a departureTime of now-or-later). So instead of faking a
// "history" scrubber, this queries the Routes API for a real predicted
// traffic-aware travel time at the *next* occurrence of Morning/Evening
// peak, using routingPreference: TRAFFIC_AWARE_OPTIMAL + trafficModel. The
// comparison against Google's traffic-free staticDuration gives a genuine
// delay/severity readout for the current origin → destination corridor —
// clearly labeled as a Google-predicted forecast, not a replay of history.
// ---------------------------------------------------------------------
const FORECAST_OPTIONS = ["Now", "Morning Peak", "Evening Peak"];
const FORECAST_TIMES = {
  "Morning Peak": { hour: 9, minute: 0 },
  "Evening Peak": { hour: 18, minute: 30 },
};
const IST_OFFSET_MINUTES = 5.5 * 60;

// Returns a real Date instant for the next occurrence of the given
// hour:minute in India Standard Time (Mumbai has no DST, fixed UTC+5:30).
function nextIstDeparture(hour, minute) {
  const istNowMs = Date.now() + IST_OFFSET_MINUTES * 60000;
  const istNow = new Date(istNowMs);
  const istTarget = new Date(
    Date.UTC(
      istNow.getUTCFullYear(),
      istNow.getUTCMonth(),
      istNow.getUTCDate(),
      hour,
      minute,
      0,
    ),
  );
  if (istTarget.getTime() <= istNowMs) {
    istTarget.setUTCDate(istTarget.getUTCDate() + 1);
  }
  return new Date(istTarget.getTime() - IST_OFFSET_MINUTES * 60000);
}

// Derives a severity label from how much slower the predicted trip is
// versus Google's traffic-free baseline for the same route — real ratio,
// not an invented number.
function severityFromDelayRatio(delayMinutes, staticDurationSeconds) {
  const staticMinutes = staticDurationSeconds / 60;
  const ratio = staticMinutes > 0 ? delayMinutes / staticMinutes : 0;
  if (ratio >= 0.6) return "SEVERE";
  if (ratio >= 0.35) return "HIGH";
  if (ratio >= 0.15) return "MODERATE";
  return "LOW";
}

const MAP_TYPE_OPTIONS = [
  { id: "roadmap", label: "Map" },
  { id: "satellite", label: "Satellite" },
  { id: "hybrid", label: "Hybrid" },
  { id: "terrain", label: "Terrain" },
];

export default function LiveTraffic() {
  const mapRef = useRef(null);
  const renderersRef = useRef([]);
  const scenarioTimerRef = useRef(null);
  const scenarioStepRef = useRef(-1);
  const remoteSyncBusyRef = useRef(false);
  const lastRemoteSignatureRef = useRef("");

  // Map markers / info window — kept in refs so we can clean them up
  // explicitly instead of leaking Google Maps overlay objects.
  const markersRef = useRef([]);
  const selectedMarkerRef = useRef(null);
  const infoWindowRef = useRef(null);
  const trafficLayerRef = useRef(null);
  const forecastRenderersRef = useRef([]);
  const historicalRenderersRef = useRef([]);

  // Traffic Agent Log: internal-only auto-scroll (never the page).
  const logScrollRef = useRef(null);
  const [autoScrollLog, setAutoScrollLog] = useState(true);
  const [hasNewLogActivity, setHasNewLogActivity] = useState(false);

  // Map controls
  const [mapTypeId, setMapTypeId] = useState("roadmap");
  const [trafficOn, setTrafficOn] = useState(true);

  // Peak-time traffic forecast (real Google Routes API prediction for the
  // current origin → destination corridor — see FORECAST_OPTIONS above).
  const [forecastMode, setForecastMode] = useState("Now");
  const [forecastResult, setForecastResult] = useState(null);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [forecastError, setForecastError] = useState(null);

  // Traffic view modes:
  // LIVE = existing Google TrafficLayer
  // PREDICTED = future Google traffic-aware route rendered on the map
  // HISTORICAL = archived Google traffic observations captured by our backend
  const [trafficViewMode, setTrafficViewMode] = useState("LIVE");
  const [historyDate, setHistoryDate] = useState(() => {
    const d = new Date();
    d.setDate(d.getDate() - 1);
    return d.toISOString().slice(0, 10);
  });
  const [historyTime, setHistoryTime] = useState("18:30");
  const [historicalTraffic, setHistoricalTraffic] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState(null);

  const { incidents } = useMapStore();

  const [mapObj, setMapObj] = useState(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [mapError, setMapError] = useState(() =>
    import.meta.env.VITE_GOOGLE_MAPS_API_KEY
      ? null
      : "VITE_GOOGLE_MAPS_API_KEY is missing.",
  );

  // Dashboard states
  const [selectedNode, setSelectedNode] = useState(null);
  const [agentDecision, setAgentDecision] = useState(null);
  const [agentStatus, setAgentStatus] = useState(
    "Select a location from High Traffic Areas...",
  );
  const [agentMetrics, setAgentMetrics] = useState(EMPTY_METRICS);

  // ---------------------------------------------------------------------
  // DISPLAY-ONLY SCENARIO STATE
  // Live Traffic is never a controller — these are purely local, visual
  // state used to render whatever SimulateDemonstation (via the backend
  // playback-state) says is currently active. They are never posted back
  // to the backend and never shared with the Zustand controller store.
  // ---------------------------------------------------------------------
  const [scenarios, setScenarios] = useState([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState("");
  const [selectedScenario, setSelectedScenario] = useState(null);
  const [scenarioStep, setScenarioStep] = useState(-1);
  const [isScenarioRunning, setIsScenarioRunning] = useState(false);
  const [isScenarioPaused, setIsScenarioPaused] = useState(false);

  const [demoTraffic, setDemoTraffic] = useState({
    locations: [],
    source: "DEMONSTRATION DATA",
  });

  const [origin, setOrigin] = useState("Dadar Station, Mumbai");
  const [destination, setDestination] = useState("Wadala, Mumbai");
  const [aiRoutes, setAiRoutes] = useState([]);
  const [isRouting, setIsRouting] = useState(false);

  // Initialize DB Data
  useEffect(() => {
    const fetchDemoData = async () => {
      try {
        const res = await fetch("http://localhost:5000/api/traffic/live");
        if (res.ok) setDemoTraffic(await res.json());

        const metricsResponse = await fetch(
          "http://localhost:5000/api/traffic/metrics",
        );
        if (metricsResponse.ok) setAgentMetrics(await metricsResponse.json());

        const scenariosResponse = await fetch(
          "http://localhost:5000/api/traffic/scenarios",
        );
        if (scenariosResponse.ok) {
          const scenarioData = await scenariosResponse.json();
          setScenarios(scenarioData.scenarios || []);
        }
      } catch (err) {
        console.error("Failed to fetch demo traffic data:", err);
      }
    };
    fetchDemoData();
    const interval = setInterval(fetchDemoData, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => () => clearTimeout(scenarioTimerRef.current), []);

  const selectTrafficNode = async (node) => {
    setSelectedNode(node);
    setAgentDecision(null);
    setAgentStatus("ANALYZING");
    try {
      const response = await fetch(
        "http://localhost:5000/api/traffic/condition",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            location: node.name,
            severity: node.severity,
            metadata: {
              trafficLevel: node.severity,
              signalId: node.signal?.signalId,
              currentGreenTime: node.signal?.currentGreenTime,
              currentRedTime: node.signal?.currentRedTime,
              source: demoTraffic.source || "DEMONSTRATION",
            },
          }),
        },
      );
      const data = await response.json();
      if (!response.ok)
        throw new Error(data.error || "Traffic analysis failed");
      setAgentDecision(data.decision?.metadata || null);
      setAgentStatus(data.status === "NO_CHANGE" ? "MONITORING" : data.status);
    } catch (error) {
      console.error("Traffic intelligence analysis failed:", error);
      setAgentStatus("ANALYSIS UNAVAILABLE");
    }
  };

  const scenarioColor = (severity) =>
    severity === "CRITICAL" || severity === "SEVERE"
      ? "text-red-500"
      : severity === "HIGH"
        ? "text-orange-400"
        : severity === "MODERATE"
          ? "text-yellow-400"
          : "text-emerald-400";

  // ---------------------------------------------------------------------
  // DEFAULT / LIVE DATASET SELECTION
  // Use genuine backend telemetry when it's available; otherwise fall back
  // to the hardcoded Mumbai demo dataset so the page is never empty. Either
  // way this is clearly labeled — never presented as real-time data unless
  // it actually is.
  // ---------------------------------------------------------------------
  const liveNodesAvailable =
    Array.isArray(demoTraffic.locations) && demoTraffic.locations.length > 0;
  const highTrafficNodes = liveNodesAvailable
    ? demoTraffic.locations
    : DEFAULT_MUMBAI_LOCATIONS;

  const telemetrySourceLabel = !liveNodesAvailable
    ? "DEMO TELEMETRY · FALLBACK"
    : demoTraffic.source === "DEMONSTRATION"
      ? "DEMO TELEMETRY"
      : "LIVE TELEMETRY";

  const isScenarioActive = !!selectedScenarioId;

  // Representative location shown in the dashboard cards when nothing has
  // been clicked yet — the highest-severity node in the current dataset.
  const defaultFocusNode = useMemo(() => {
    if (!highTrafficNodes.length) return null;
    return [...highTrafficNodes].sort(
      (a, b) => (SEVERITY_RANK[a.severity] ?? 9) - (SEVERITY_RANK[b.severity] ?? 9),
    )[0];
  }, [highTrafficNodes]);

  const activeNode = selectedNode || defaultFocusNode;
  const isDefaultFocus = !selectedNode;

  const isDefaultSignal = !agentDecision?.signal;
  const signalSource = agentDecision?.signal
    ? agentDecision.signal
    : {
        currentGreen:
          activeNode?.signal?.currentGreenTime ??
          defaultSignalTiming(activeNode?.severity).currentGreen,
        currentRed:
          activeNode?.signal?.currentRedTime ??
          defaultSignalTiming(activeNode?.severity).currentRed,
        optimizedGreen: defaultSignalTiming(activeNode?.severity).optGreen,
        optimizedRed: defaultSignalTiming(activeNode?.severity).optRed,
        greenAdjustment: null,
      };

  const handleSelectNode = (node) => {
    clearTimeout(scenarioTimerRef.current);
    scenarioStepRef.current = -1;
    setSelectedScenarioId("");
    setSelectedScenario(null);
    setScenarioStep(-1);
    setIsScenarioRunning(false);
    setIsScenarioPaused(false);
    selectTrafficNode(node);
  };

  const scenarioContent = useMemo(() => {
    if (!selectedScenario) return null;
    const name = selectedScenario.name.toLowerCase();

    let content = {
      event: "Evening peak traffic demand increases.",
      analysis: "High demand on Dadar → Wadala.",
      prediction: "Congestion expected to deteriorate to SEVERE.",
      decision: "SIGNAL OPTIMIZATION",
      action: "Optimize signal timing and evaluate alternative routes.",
      impact: "Reduce traffic buildup and balance network flow.",
      outcomeSignal: "38s GREEN",
      signalDiff: {
        greenDiff: "+8s",
        redDiff: "-8s",
        currentGreen: 30,
        currentRed: 40,
        optGreen: 38,
        optRed: 32,
      },
      outcomeRoute: "ROUTE B",
    };

    if (name.includes("accident")) {
      content = {
        event: "Accident detected on corridor.",
        analysis: "Lane blockage detected on primary route.",
        prediction: "Rapid bottleneck; gridlock imminent.",
        decision: "ROUTE DIVERSION",
        action: "Recommend diversion, coordinate traffic response.",
        impact: "Prevent secondary collisions and clear bottleneck.",
        outcomeSignal: "ADAPTIVE",
        signalDiff: {
          greenDiff: "+5s",
          redDiff: "-5s",
          currentGreen: 40,
          currentRed: 40,
          optGreen: 45,
          optRed: 35,
        },
        outcomeRoute: "DIVERSION ROUTE",
      };
    } else if (name.includes("ambulance") || name.includes("green corridor")) {
      content = {
        event: "Critical patient transport request received.",
        analysis: "Evaluating intersections along emergency route.",
        prediction: "Standard routing would cause fatal delays.",
        decision: "EMERGENCY PRIORITY",
        action: "Prioritize emergency corridor (Simulation).",
        impact: "Uninterrupted ambulance progression.",
        outcomeSignal: "GREEN WAVE",
        signalDiff: {
          greenDiff: "+20s",
          redDiff: "-20s",
          currentGreen: 30,
          currentRed: 50,
          optGreen: 50,
          optRed: 30,
        },
        outcomeRoute: "EMERGENCY CORRIDOR",
      };
    } else if (name.includes("closure")) {
      content = {
        event: "Road section became unexpectedly unavailable.",
        analysis: "Identify closure, evaluate alternatives.",
        prediction: "High traffic density shifting to adjacent roads.",
        decision: "ROAD DIVERSION",
        action: "Recommend alternative route and divert traffic.",
        impact: "Safe rerouting of civilian traffic.",
        outcomeSignal: "MAINTAINED",
        signalDiff: {
          greenDiff: "0s",
          redDiff: "0s",
          currentGreen: 40,
          currentRed: 40,
          optGreen: 40,
          optRed: 40,
        },
        outcomeRoute: "ALTERNATIVE ROUTE",
      };
    } else if (
      name.includes("surge") ||
      name.includes("bkc") ||
      name.includes("event")
    ) {
      content = {
        event: "Large event expected to increase traffic.",
        analysis: "Analyze expected demand from event egress.",
        prediction: "Severe congestion forecasted across exit nodes.",
        decision: "PROACTIVE OPTIMIZATION",
        action: "Proactive signal optimization and alternate routing.",
        impact: "Controlled traffic dispersion and reduced peak delay.",
        outcomeSignal: "PREEMPTIVE (+10s)",
        signalDiff: {
          greenDiff: "+10s",
          redDiff: "-10s",
          currentGreen: 35,
          currentRed: 45,
          optGreen: 45,
          optRed: 35,
        },
        outcomeRoute: "DISPERSAL ROUTE",
      };
    } else if (
      name.includes("waterlog") ||
      name.includes("monsoon") ||
      name.includes("flood")
    ) {
      content = {
        event: "Heavy rainfall causing severe waterlogging.",
        analysis: "Assess affected road section for transit safety.",
        prediction: "High risk of stranded vehicles.",
        decision: "ROAD RESTRICTION + DIVERSION",
        action: "Restrict route, recommend safe alternative.",
        impact: "Prevent vehicle damage and safely guide traffic.",
        outcomeSignal: "RESTRICTED",
        signalDiff: {
          greenDiff: "-15s",
          redDiff: "+15s",
          currentGreen: 45,
          currentRed: 35,
          optGreen: 30,
          optRed: 50,
        },
        outcomeRoute: "SAFE ELEVATED ROUTE",
      };
    }
    return content;
  }, [selectedScenario]);

  const terminalLogs = useMemo(() => {
    if (!selectedScenario || scenarioStep < 0) return [];
    const timeStr = new Date().toLocaleTimeString([], { hour12: false });
    const logs = [];
    if (scenarioStep >= 0)
      logs.push(`[${timeStr}] Traffic telemetry received.`);
    if (scenarioStep >= 1) {
      logs.push(
        `[${timeStr}] ${selectedScenario.location} classified as ${selectedScenario.severity}.`,
      );
      logs.push(`[${timeStr}] Analysis: ${scenarioContent.analysis}`);
    }
    if (scenarioStep >= 2)
      logs.push(`[${timeStr}] Prediction: ${scenarioContent.prediction}`);
    if (scenarioStep >= 3)
      logs.push(`[${timeStr}] Decision formed: ${scenarioContent.decision}`);
    if (scenarioStep >= 4) {
      logs.push(`[${timeStr}] Evaluating signal timing limits.`);
      logs.push(
        `[${timeStr}] Recommendation generated: GREEN ${scenarioContent.signalDiff.greenDiff}`,
      );
    }
    if (scenarioStep >= 5) {
      logs.push(`[${timeStr}] Evaluating alternate routes.`);
      logs.push(
        `[${timeStr}] Route recommended: ${scenarioContent.outcomeRoute}`,
      );
    }
    if (scenarioStep >= 6) {
      logs.push(`[${timeStr}] Decision recorded.`);
      logs.push(`[${timeStr}] End of simulation loop.`);
    }
    return logs;
  }, [scenarioStep, selectedScenario, scenarioContent]);

  // ---------------------------------------------------------------------
  // TRAFFIC AGENT LOG — internal scroll only, never the page.
  // Auto-scrolls the log container itself (via scrollTop) only while the
  // user is already near its bottom; if they've scrolled up to read
  // earlier entries, new logs arrive quietly with a "new activity" nudge
  // instead of yanking them back down. The page's own scroll position is
  // never touched here.
  // ---------------------------------------------------------------------
  const handleLogScroll = () => {
    const el = logScrollRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const nearBottom = distanceFromBottom < 24;
    setAutoScrollLog(nearBottom);
    if (nearBottom) setHasNewLogActivity(false);
  };

  const scrollLogToBottom = () => {
    const el = logScrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    setAutoScrollLog(true);
    setHasNewLogActivity(false);
  };

  useEffect(() => {
    const el = logScrollRef.current;
    if (!el) return;
    if (autoScrollLog) {
      el.scrollTop = el.scrollHeight;
    } else if (terminalLogs.length > 0) {
      setHasNewLogActivity(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [terminalLogs]);

  // Initialize Map
  useEffect(() => {
    const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
    if (!apiKey) return;

    const initMap = async () => {
      try {
        if (!mapRef.current) return;
        const { Map, TrafficLayer } =
          await window.google.maps.importLibrary("maps");

        const initialMap = new Map(mapRef.current, {
          center: { lat: 19.0178, lng: 72.8478 },
          zoom: 14,
          backgroundColor: "#0f172a",
          disableDefaultUI: false,
          mapTypeControl: true,
          styles: [
            { elementType: "geometry", stylers: [{ color: "#1e293b" }] },
            {
              elementType: "labels.text.stroke",
              stylers: [{ color: "#0f172a" }],
            },
            {
              elementType: "labels.text.fill",
              stylers: [{ color: "#94a3b8" }],
            },
            {
              featureType: "road",
              elementType: "geometry",
              stylers: [{ color: "#334155" }],
            },
            {
              featureType: "road.highway",
              elementType: "geometry",
              stylers: [{ color: "#475569" }],
            },
            {
              featureType: "water",
              elementType: "geometry",
              stylers: [{ color: "#020617" }],
            },
          ],
        });

        const trafficLayer = new TrafficLayer();
        trafficLayer.setMap(initialMap);
        trafficLayerRef.current = trafficLayer;
        setMapObj(initialMap);
        setMapLoaded(true);
      } catch (err) {
        console.error("Failed to load Google Maps internal libraries:", err);
        setMapError("Failed to initialize Google Maps.");
      }
    };

    const scriptId = "google-maps-script-loader";
    const existingScript = document.getElementById(scriptId);

    if (!existingScript && !window.google) {
      window.initGoogleMapsCallback = () => {
        initMap();
      };
      const script = document.createElement("script");
      script.id = scriptId;
      script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&loading=async&callback=initGoogleMapsCallback`;
      script.async = true;
      script.defer = true;
      script.onerror = () => setMapError("Failed to load Google Maps script.");
      document.head.appendChild(script);
    } else if (window.google && window.google.maps) {
      initMap();
    }
  }, []);

  const clearRenderers = () => {
    renderersRef.current.forEach((polyline) => polyline.setMap(null));
    renderersRef.current = [];
  };


  const clearOverlayRenderers = (ref) => {
    ref.current.forEach((polyline) => polyline.setMap(null));
    ref.current = [];
  };

  const trafficSpeedColor = (speed) => {
    switch (speed) {
      case "TRAFFIC_JAM":
        return "#ef4444";
      case "SLOW":
        return "#f59e0b";
      case "NORMAL":
      default:
        return "#22c55e";
    }
  };

  // Draw Google's traffic-aware polyline intervals. Google returns indexes
  // into the decoded polyline rather than separate geometry for each interval.
  const drawTrafficAwareRoute = async (route, targetRef, zIndex = 120) => {
    if (!mapObj || !window.google || !route?.polyline) return [];

    const { encoding } =
      await window.google.maps.importLibrary("geometry");
    const { Polyline } =
      await window.google.maps.importLibrary("maps");

    const path = encoding.decodePath(route.polyline);
    const intervals = route.speedReadingIntervals || [];

    if (!intervals.length) {
      const line = new Polyline({
        path,
        map: mapObj,
        strokeColor: "#3b82f6",
        strokeWeight: 7,
        strokeOpacity: 0.9,
        zIndex,
      });
      targetRef.current.push(line);
      return [line];
    }

    const lines = intervals
      .map((interval) => {
        const start = Number.isFinite(interval.startPolylinePointIndex)
          ? interval.startPolylinePointIndex
          : 0;
        const end = Number.isFinite(interval.endPolylinePointIndex)
          ? interval.endPolylinePointIndex
          : path.getLength() - 1;

        if (end <= start) return null;

        const segment = [];
        for (let index = start; index <= end; index += 1) {
          segment.push(path.getAt(index));
        }

        if (segment.length < 2) return null;

        const line = new Polyline({
          path: segment,
          map: mapObj,
          strokeColor: trafficSpeedColor(interval.speed),
          strokeWeight: 8,
          strokeOpacity: 0.95,
          zIndex,
        });

        targetRef.current.push(line);
        return line;
      })
      .filter(Boolean);

    return lines;
  };

  // MODERN ROUTES API IMPLEMENTATION (kept internally to power "Best Route" in AI Predictions;
  // the dedicated Route Intelligence card has been removed from the UI per request)
  const calculateAiRoute = async (
    e,
    org = origin,
    dest = destination,
    scenarioObj = selectedScenario,
  ) => {
    if (e) e.preventDefault();
    if (!mapObj || !org || !dest) return;

    setIsRouting(true);
    clearRenderers();
    setMapError(null);

    try {
      const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
      const normalizedRoutes = await calculateModernRoutes(org, dest, apiKey);

      if (normalizedRoutes.length === 0) {
        setIsRouting(false);
        return;
      }

      const { encoding } = await window.google.maps.importLibrary("geometry");
      const { Polyline } = await window.google.maps.importLibrary("maps");
      const { LatLngBounds } = await window.google.maps.importLibrary("core");

      const bounds = new LatLngBounds();

      const evaluatedRoutes = normalizedRoutes.map((route, index) => {
        const path = encoding.decodePath(route.polyline);
        path.forEach((latLng) => bounds.extend(latLng));

        const polyline = new Polyline({
          path: path,
          map: mapObj,
          strokeColor: "#64748b",
          strokeWeight: 4,
          strokeOpacity: 0.6,
          zIndex: 10,
        });
        renderersRef.current.push(polyline);

        return {
          id: index,
          name: route.name,
          summary: route.summary,
          distance: route.distance,
          duration: route.duration,
          recommended: false,
          incident: false,
          impact: "UNKNOWN",
          reason: "Awaiting Traffic Agent evaluation.",
          trafficLevel: selectedNode?.severity || "MODERATE",
          predictedTrafficLevel: selectedNode?.severity || "MODERATE",
          durationSeconds: route.durationSeconds,
        };
      });

      mapObj.fitBounds(bounds);

      fetch("http://localhost:5000/api/traffic/routes/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          origin: org,
          destination: dest,
          location: scenarioObj?.location || selectedNode?.name || org,
          routes: evaluatedRoutes,
          source: demoTraffic.source || "DEMONSTRATION",
          scenario: scenarioObj,
        }),
      })
        .then(async (response) => {
          const data = await response.json();
          if (!response.ok) throw new Error("Route evaluation failed");

          const recommendation = data.decision?.metadata?.routeRecommendation;
          const impacted = recommendation?.affectedRoutes || [];

          renderersRef.current.forEach((polyline, index) => {
            const routeName = evaluatedRoutes[index].name;
            const isRec =
              evaluatedRoutes[index].id === recommendation?.id ||
              routeName === recommendation?.name;
            const hasInc = impacted.includes(routeName);

            polyline.setOptions({
              strokeColor: isRec ? "#3b82f6" : hasInc ? "#ef4444" : "#64748b",
              strokeWeight: isRec ? 7 : 4,
              strokeOpacity: isRec ? 1.0 : 0.6,
              zIndex: isRec ? 100 : 10,
            });
          });

          setAiRoutes(
            evaluatedRoutes.map((route) => ({
              ...route,
              recommended:
                route.id === recommendation?.id ||
                route.name === recommendation?.name,
              incident: impacted.includes(route.name),
              reason:
                route.name === recommendation?.name
                  ? recommendation.reason
                  : impacted.includes(route.name)
                    ? "Affected by an active incident/congestion supplied to the agent."
                    : "Higher predicted traffic impact than the recommended route.",
            })),
          );
        })
        .catch((err) => {
          console.error("Agent evaluation failed:", err);
          setAiRoutes(evaluatedRoutes);
        });
    } catch (error) {
      setMapError(error.message);
    } finally {
      setIsRouting(false);
    }
  };

  useEffect(() => {
    if (mapLoaded && selectedScenario) {
      calculateAiRoute(null, origin, destination, selectedScenario);
    }
  }, [mapLoaded]);

  // ---------------------------------------------------------------------
  // PEAK-TIME FORECAST FETCH + MAP OVERLAY
  // ---------------------------------------------------------------------
  useEffect(() => {
    clearOverlayRenderers(forecastRenderersRef);

    if (trafficViewMode !== "PREDICTED" || forecastMode === "Now") {
      setForecastResult(null);
      setForecastError(null);
      setForecastLoading(false);
      return;
    }

    const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
    if (!apiKey || !origin || !destination || !mapObj) return;

    const timing = FORECAST_TIMES[forecastMode];
    if (!timing) return;
    const departure = nextIstDeparture(timing.hour, timing.minute);

    let cancelled = false;
    setForecastLoading(true);
    setForecastError(null);

    calculateModernRoutes(origin, destination, apiKey, {
      departureTime: departure.toISOString(),
    })
      .then(async (routes) => {
        if (cancelled) return;

        const best = routes[0] || null;
        if (!best) {
          setForecastResult(null);
          setForecastError("No predicted route available for this corridor.");
          return;
        }

        setForecastResult({ route: best, departureAt: departure });

        await drawTrafficAwareRoute(
          best,
          forecastRenderersRef,
          140,
        );
      })
      .catch((err) => {
        if (cancelled) return;
        console.error("Peak-time traffic forecast failed:", err);
        setForecastResult(null);
        setForecastError(
          err.message || "Traffic forecast unavailable.",
        );
      })
      .finally(() => {
        if (!cancelled) setForecastLoading(false);
      });

    return () => {
      cancelled = true;
      clearOverlayRenderers(forecastRenderersRef);
    };
  }, [
    trafficViewMode,
    forecastMode,
    origin,
    destination,
    mapObj,
  ]);

  // ---------------------------------------------------------------------
  // HISTORICAL TRAFFIC FETCH + MAP OVERLAY
  // Historical data is ONLY whatever the backend has archived. We never
  // generate a fake past Google traffic state.
  // ---------------------------------------------------------------------
  useEffect(() => {
    clearOverlayRenderers(historicalRenderersRef);

    if (trafficViewMode !== "HISTORICAL" || !mapObj) {
      setHistoricalTraffic([]);
      setHistoryError(null);
      setHistoryLoading(false);
      return;
    }

    let cancelled = false;
    setHistoryLoading(true);
    setHistoryError(null);

    fetch(
      `http://localhost:5000/api/traffic/history?date=${encodeURIComponent(
        historyDate,
      )}&time=${encodeURIComponent(historyTime)}`,
    )
      .then(async (response) => {
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(
            data.error || "Historical traffic request failed.",
          );
        }
        return data;
      })
      .then(async (data) => {
        if (cancelled) return;

        const locations = data.locations || [];
        setHistoricalTraffic(locations);

        if (!locations.length) {
          setHistoryError(
            "No archived Google traffic observations exist for this date/time.",
          );
          return;
        }

        const { encoding } =
          await window.google.maps.importLibrary("geometry");
        const { Polyline } =
          await window.google.maps.importLibrary("maps");

        locations.forEach((observation) => {
          if (!observation.encoded_polyline) return;

          const path = encoding.decodePath(
            observation.encoded_polyline,
          );

          const color =
            observation.speed_category === "TRAFFIC_JAM"
              ? "#ef4444"
              : observation.speed_category === "SLOW"
                ? "#f59e0b"
                : observation.speed_category === "NORMAL"
                  ? "#22c55e"
                  : observation.traffic_level === "SEVERE" ||
                      observation.traffic_level === "HIGH"
                    ? "#ef4444"
                    : observation.traffic_level === "MODERATE"
                      ? "#f59e0b"
                      : "#22c55e";

          const line = new Polyline({
            path,
            map: mapObj,
            strokeColor: color,
            strokeWeight: 7,
            strokeOpacity: 0.9,
            zIndex: 130,
          });

          historicalRenderersRef.current.push(line);
        });
      })
      .catch((err) => {
        if (cancelled) return;
        console.error("Historical traffic load failed:", err);
        setHistoricalTraffic([]);
        setHistoryError(
          err.message || "Historical traffic unavailable.",
        );
      })
      .finally(() => {
        if (!cancelled) setHistoryLoading(false);
      });

    return () => {
      cancelled = true;
      clearOverlayRenderers(historicalRenderersRef);
    };
  }, [trafficViewMode, historyDate, historyTime, mapObj]);


  // Loads a scenario's data for display only (GET only — never writes
  // playback-state, never triggers a replay). Called only in response to
  // what the remote-sync poll below observes.
  const prepareScenario = async (scenarioId) => {
    clearTimeout(scenarioTimerRef.current);
    scenarioStepRef.current = -1;
    setIsScenarioRunning(false);
    setIsScenarioPaused(false);
    setScenarioStep(-1);
    setAgentDecision(null);
    setAiRoutes([]);
    clearRenderers();
    setAgentStatus("SCENARIO READY");

    if (!scenarioId) {
      setSelectedScenarioId("");
      setSelectedScenario(null);
      setSelectedNode(null);
      return;
    }

    try {
      const response = await fetch(
        `http://localhost:5000/api/traffic/scenarios/${scenarioId}`,
      );
      const data = await response.json();
      if (!response.ok)
        throw new Error(data.error || "Failed to load scenario");
      const scenario = data.scenario;

      setSelectedScenarioId(String(scenario.id));
      setSelectedScenario(scenario);
      setSelectedNode({
        id: `scenario-${scenario.id}`,
        name: scenario.location,
        lat: scenario.coordinates.lat,
        lng: scenario.coordinates.lng,
        severity: scenario.severity,
        color: scenarioColor(scenario.severity),
        scenario,
      });

      let newOrg = "Dadar Station, Mumbai";
      let newDest = "Wadala, Mumbai";

      const scenarioName = scenario.name.toLowerCase();
      if (scenarioName.includes("accident")) {
        newDest = "Sion, Mumbai";
      } else if (scenarioName.includes("ambulance")) {
        newOrg = "Dadar, Mumbai";
        newDest = "KEM Hospital, Parel, Mumbai";
      } else if (scenarioName.includes("closure")) {
        newOrg = "Sion, Mumbai";
        newDest = "Wadala, Mumbai";
      } else if (scenarioName.includes("bkc")) {
        newOrg = "Bandra Station, Mumbai";
        newDest = "BKC, Mumbai";
      } else if (scenarioName.includes("monsoon")) {
        newOrg = "Sion, Mumbai";
        newDest = "Wadala, Mumbai";
      }

      setOrigin(newOrg);
      setDestination(newDest);

      if (mapObj) {
        calculateAiRoute(null, newOrg, newDest, scenario);
      }
    } catch (error) {
      console.error("Loading scenario for display failed:", error);
      setAgentStatus("SCENARIO UNAVAILABLE");
    }
  };

  // ---------------------------------------------------------------------
  // LOCAL DISPLAY ANIMATION ONLY
  // Steps the DETECT → OUTCOME visual sequence for whichever scenario is
  // currently loaded. Triggered only by what the remote-sync poll below
  // observes — never by a local control — and never calls /replay or
  // writes playback-state back to the server. Purely cosmetic.
  // ---------------------------------------------------------------------
  const startLocalAnimation = () => {
    if (!selectedScenarioId || isScenarioRunning) return;
    if (scenarioStepRef.current >= SCENARIO_STEPS.length - 1) {
      scenarioStepRef.current = -1;
      setScenarioStep(-1);
    }
    setIsScenarioRunning(true);
    setIsScenarioPaused(false);

    if (aiRoutes.length === 0) {
      calculateAiRoute(null, origin, destination, selectedScenario);
    }

    const advance = () => {
      const nextStep = scenarioStepRef.current + 1;
      scenarioStepRef.current = nextStep;
      setScenarioStep(nextStep);

      if (nextStep < SCENARIO_STEPS.length - 1) {
        scenarioTimerRef.current = setTimeout(advance, 1500);
        return;
      }
      setAgentStatus("SCENARIO COMPLETE");
      setIsScenarioRunning(false);
    };
    advance();
  };

  const pauseLocalAnimation = () => {
    clearTimeout(scenarioTimerRef.current);
    setIsScenarioRunning(false);
    setIsScenarioPaused(true);
  };

  // ---------------------------------------------------------------------
  // CROSS-DEVICE PLAYBACK MIRROR (read-only)
  // Live Traffic never selects a scenario or starts/pauses/resets on its
  // own. It only polls the backend for whatever SimulateDemonstation (on
  // this device or any other) last set via Zustand → playback-state, and
  // mirrors that here visually. It never POSTs playback-state and never
  // calls /replay — this effect only reads.
  // ---------------------------------------------------------------------
  useEffect(() => {
    let cancelled = false;

    const syncFromRemote = async () => {
      if (remoteSyncBusyRef.current) return;
      try {
        const res = await fetch(
          "http://localhost:5000/api/traffic/playback-state",
        );
        if (!res.ok || cancelled) return;
        const remote = await res.json();

        const remoteScenarioId = remote?.scenarioId
          ? String(remote.scenarioId)
          : "";
        const remoteRunning = !!remote?.isRunning;
        const remotePaused = !!remote?.isPaused;

        // Skip re-processing the exact same remote state repeatedly
        const signature = `${remoteScenarioId}|${remoteRunning}|${remotePaused}`;
        if (signature === lastRemoteSignatureRef.current) return;
        lastRemoteSignatureRef.current = signature;

        remoteSyncBusyRef.current = true;

        // Remote cleared the scenario (Reset to Manual Exploration Mode)
        if (!remoteScenarioId) {
          if (selectedScenarioId) {
            await prepareScenario("");
          }
          return;
        }

        // Remote selected/started a different scenario than what's loaded here
        if (remoteScenarioId !== selectedScenarioId) {
          await prepareScenario(remoteScenarioId);
          if (remoteRunning) startLocalAnimation();
          return;
        }

        // Same scenario already loaded — mirror Start/Pause/Reset state
        if (remoteRunning && !isScenarioRunning) {
          startLocalAnimation();
        } else if (!remoteRunning && remotePaused && isScenarioRunning) {
          pauseLocalAnimation();
        } else if (
          !remoteRunning &&
          !remotePaused &&
          (isScenarioRunning || isScenarioPaused)
        ) {
          // Remote pressed Reset on the same scenario
          await prepareScenario(remoteScenarioId);
        }
      } catch (err) {
        // Backend may be briefly unreachable — stay silent and retry on next poll
      } finally {
        remoteSyncBusyRef.current = false;
      }
    };

    syncFromRemote();
    const pollInterval = setInterval(syncFromRemote, 2000);
    return () => {
      cancelled = true;
      clearInterval(pollInterval);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedScenarioId, isScenarioRunning, isScenarioPaused]);

  // Map type control
  useEffect(() => {
    if (mapObj) mapObj.setMapTypeId(mapTypeId);
  }, [mapObj, mapTypeId]);

  // Traffic layer toggle (reuses the same TrafficLayer created at init)
  useEffect(() => {
    if (trafficLayerRef.current) {
      trafficLayerRef.current.setMap(
        trafficOn && trafficViewMode === "LIVE" ? mapObj : null,
      );
    }
  }, [trafficOn, trafficViewMode, mapObj]);

  // ---------------------------------------------------------------------
  // ALL-LOCATIONS MARKERS
  // Small clickable dots for every location currently in view (live or
  // default dataset). Cleared and rebuilt whenever the dataset changes —
  // never accumulates duplicate markers.
  // ---------------------------------------------------------------------
  useEffect(() => {
    if (!mapObj || !window.google) return undefined;

    const markers = highTrafficNodes.map((node) => {
      const lat = parseFloat(node.lat);
      const lng = parseFloat(node.lng);
      if (Number.isNaN(lat) || Number.isNaN(lng)) return null;

      const marker = new window.google.maps.Marker({
        position: { lat, lng },
        map: mapObj,
        title: `${node.name} — ${node.severity}`,
        icon: {
          path: window.google.maps.SymbolPath.CIRCLE,
          scale: 6,
          fillColor: severityMarkerHex(node.severity),
          fillOpacity: 0.85,
          strokeColor: "#0f172a",
          strokeWeight: 1.5,
        },
      });
      marker.addListener("click", () => handleSelectNode(node));
      return marker;
    }).filter(Boolean);

    markersRef.current = markers;

    return () => {
      markers.forEach((marker) => marker.setMap(null));
      markersRef.current = [];
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mapObj, highTrafficNodes]);

  // ---------------------------------------------------------------------
  // SELECTED-LOCATION MARKER + INFO WINDOW
  // Only the currently selected location gets the highlighted marker and
  // the map pan/zoom — clicking a location is a genuine map interaction,
  // not just a camera move. One marker/info-window at a time; the old one
  // is always cleaned up first.
  // ---------------------------------------------------------------------
  useEffect(() => {
    if (!mapObj || !window.google) return;

    if (selectedMarkerRef.current) {
      selectedMarkerRef.current.setMap(null);
      selectedMarkerRef.current = null;
    }
    if (infoWindowRef.current) {
      infoWindowRef.current.close();
    }
    if (!selectedNode) return;

    const lat = parseFloat(selectedNode.lat);
    const lng = parseFloat(selectedNode.lng);
    if (Number.isNaN(lat) || Number.isNaN(lng)) return;

    const marker = new window.google.maps.Marker({
      position: { lat, lng },
      map: mapObj,
      title: selectedNode.name,
      zIndex: 999,
      icon: {
        path: window.google.maps.SymbolPath.CIRCLE,
        scale: 10,
        fillColor: severityMarkerHex(selectedNode.severity),
        fillOpacity: 1,
        strokeColor: "#0f172a",
        strokeWeight: 3,
      },
      animation: window.google.maps.Animation.DROP,
    });
    selectedMarkerRef.current = marker;

    if (!infoWindowRef.current) {
      infoWindowRef.current = new window.google.maps.InfoWindow();
    }
    const speedLine =
      selectedNode.averageSpeed != null
        ? `<div style="font-size:11px;color:#475569;">Avg speed: <strong>${selectedNode.averageSpeed} km/h</strong></div>`
        : "";
    const delayLine =
      selectedNode.delayMinutes != null
        ? `<div style="font-size:11px;color:#475569;">Delay: <strong>${selectedNode.delayMinutes} min</strong></div>`
        : "";
    infoWindowRef.current.setContent(
      `<div style="font-family:sans-serif;min-width:150px;">` +
        `<div style="font-weight:700;font-size:13px;margin-bottom:4px;">${selectedNode.name}</div>` +
        `<div style="font-size:11px;color:#475569;">Severity: <strong>${selectedNode.severity}</strong></div>` +
        speedLine +
        delayLine +
        `</div>`,
    );
    infoWindowRef.current.open({ map: mapObj, anchor: marker });

    mapObj.panTo({ lat, lng });
    mapObj.setZoom(16);
  }, [mapObj, selectedNode]);

  const handleCenterOnMumbai = () => {
    if (!mapObj) return;
    mapObj.panTo(MUMBAI_CENTER);
    mapObj.setZoom(MUMBAI_DEFAULT_ZOOM);
  };

  const handleFitTrafficAreas = () => {
    if (!mapObj || !window.google || highTrafficNodes.length === 0) return;
    const bounds = new window.google.maps.LatLngBounds();
    highTrafficNodes.forEach((node) => {
      const lat = parseFloat(node.lat);
      const lng = parseFloat(node.lng);
      if (!Number.isNaN(lat) && !Number.isNaN(lng)) bounds.extend({ lat, lng });
    });
    mapObj.fitBounds(bounds);
  };

  const handleFocusSelected = () => {
    if (!mapObj || !activeNode) return;
    const lat = parseFloat(activeNode.lat);
    const lng = parseFloat(activeNode.lng);
    if (Number.isNaN(lat) || Number.isNaN(lng)) return;
    if (!selectedNode) {
      handleSelectNode(activeNode);
      return;
    }
    mapObj.panTo({ lat, lng });
    mapObj.setZoom(16);
  };

  const bestRoute =
    agentDecision?.routeRecommendation?.name ||
    aiRoutes.find((r) => r.recommended)?.name ||
    "Unavailable";

  const renderSignalBlock = (approachData) => {
    if (!approachData) return null;
    const totalCurrent = approachData.currentGreen + approachData.currentRed;
    return (
      <div className="flex flex-col gap-2 mb-3">
        <div className="flex justify-between items-center text-xs">
          <span className="font-bold text-slate-400 w-16">GREEN</span>
          <div className="flex-1 mx-2 flex gap-4 items-center">
            <div className="h-2 flex-1 rounded bg-slate-800 overflow-hidden">
              <div
                className="h-full bg-emerald-600"
                style={{
                  width: `${(approachData.currentGreen / totalCurrent) * 100}%`,
                }}
              ></div>
            </div>
            <span className="font-mono text-emerald-500 w-8">
              {approachData.currentGreen}s
            </span>
          </div>
        </div>
        <div className="flex justify-between items-center text-xs">
          <span className="font-bold text-slate-400 w-16">RED</span>
          <div className="flex-1 mx-2 flex gap-4 items-center">
            <div className="h-2 flex-1 rounded bg-slate-800 overflow-hidden">
              <div
                className="h-full bg-red-600"
                style={{
                  width: `${(approachData.currentRed / totalCurrent) * 100}%`,
                }}
              ></div>
            </div>
            <span className="font-mono text-red-500 w-8">
              {approachData.currentRed}s
            </span>
          </div>
        </div>
      </div>
    );
  };

  const isReplayView = scenarioStep >= 0 && selectedScenario;

  return (
    <div className="h-full flex flex-col gap-6 text-slate-100 font-sans overflow-y-auto custom-scrollbar pr-2 pb-6">
      {/* HEADER */}
      <div className="bg-slate-900 border border-slate-700 p-4 rounded-xl flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4 shrink-0 shadow-lg">
        <div className="flex flex-col">
          <div className="flex items-center gap-3">
            <BrainIcon />
            <h1 className="text-xl font-bold text-slate-100 tracking-tight uppercase">
              Live Traffic Analysis
            </h1>
          </div>
          <p className="text-xs text-slate-400 font-medium mt-1">
            Live Intelligence Playback
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-4 w-full lg:w-auto">
          <div
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border ${isScenarioActive ? "bg-blue-900/30 border-blue-500/50" : "bg-slate-950 border-slate-700"}`}
          >
            <span
              className={`w-2.5 h-2.5 rounded-full ${isScenarioActive ? "bg-blue-400 shadow-[0_0_8px_rgba(96,165,250,0.8)] animate-pulse" : "bg-emerald-400"}`}
            ></span>
            <span
              className={`text-[10px] font-mono font-bold tracking-widest ${isScenarioActive ? "text-blue-300" : "text-slate-400"}`}
            >
              {isScenarioActive
                ? "SCENARIO PLAYBACK"
                : `LIVE TRAFFIC OVERVIEW · ${telemetrySourceLabel}`}
            </span>
          </div>

          {/* Display-only: playback is driven entirely from the Simulate
              Demonstration page. No local start/pause/reset controls here. */}
          <div className="flex items-center gap-2 bg-slate-950 border border-slate-700 px-3 py-1.5 rounded-lg">
            <span
              className={`w-2 h-2 rounded-full ${isScenarioRunning ? "bg-blue-400 animate-pulse" : "bg-slate-600"}`}
            ></span>
            <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400">
              {selectedScenarioId
                ? isScenarioRunning
                  ? "Mirroring live playback"
                  : isScenarioPaused
                    ? "Playback paused"
                    : "Scenario loaded"
                : "Waiting for Simulate Demonstration"}
            </span>
          </div>
        </div>
      </div>

      {/* MAP + HIGH TRAFFIC AREAS */}
      <section className="grid grid-cols-1 lg:grid-cols-4 gap-6 shrink-0">
        <div className="lg:col-span-1 bg-slate-900/80 border border-slate-700/50 rounded-xl shadow-lg flex flex-col overflow-hidden h-full max-h-[650px]">
          <div className="p-4 border-b border-slate-700/50 bg-slate-800/30 flex justify-between items-center">
            <h3 className="font-bold text-sm uppercase tracking-widest text-slate-200">
              High Traffic Areas
            </h3>
            <span className="text-[9px] bg-slate-950 px-2 py-0.5 rounded text-slate-400 font-mono border border-slate-800 uppercase">
              {telemetrySourceLabel}
            </span>
          </div>
          <div className="p-4 flex flex-col gap-2 overflow-y-auto custom-scrollbar flex-1">
            {highTrafficNodes.length === 0 ? (
              <div className="text-slate-500 text-sm italic text-center mt-6">
                Loading backend telemetry...
              </div>
            ) : (
              highTrafficNodes.map((node) => (
                <button
                  key={node.id}
                  onClick={() => handleSelectNode(node)}
                  title={`Focus ${node.name} on the map`}
                  className={`flex justify-between items-center p-3 rounded-lg border text-left transition-all ${selectedNode?.id === node.id ? "bg-slate-800 border-blue-500/50 shadow-inner" : "bg-slate-950/50 border-slate-800 hover:border-slate-600"}`}
                >
                  <span className="text-sm text-slate-200 font-bold truncate">
                    {node.name}
                  </span>
                  <div className="flex items-center gap-2">
                    <span
                      className={`w-2 h-2 rounded-full ${node.severity === "SEVERE" ? "bg-red-500" : node.severity === "HIGH" ? "bg-orange-500" : node.severity === "MODERATE" ? "bg-yellow-400" : "bg-emerald-400"}`}
                    ></span>
                    <span
                      className={`text-[10px] font-bold uppercase tracking-widest ${node.color}`}
                    >
                      {node.severity}
                    </span>
                  </div>
                </button>
              ))
            )}
          </div>
        </div>

        {/* Point 2/3: Map made larger, terminal log NO LONGER overlaps the map */}
        <div className="lg:col-span-3 bg-slate-900 border border-slate-700/80 rounded-xl overflow-hidden shadow-lg relative flex flex-col min-h-[500px] lg:min-h-[650px]">
          <div className="flex-1 relative bg-slate-950">
            {mapError ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-red-400 p-6 text-center z-10">
                <AlertIcon />
                <span className="mt-2 text-sm font-mono">{mapError}</span>
              </div>
            ) : !mapLoaded ? (
              <div className="absolute inset-0 flex items-center justify-center text-slate-400 z-10">
                <span className="text-sm font-mono animate-pulse">
                  Initializing map...
                </span>
              </div>
            ) : null}
            <div ref={mapRef} className="absolute inset-0 w-full h-full" />
            <div className="absolute bottom-1 left-2 z-10 bg-slate-950/80 backdrop-blur px-3 py-1.5 rounded border border-slate-700 shadow-lg text-[10px] font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
              <MapPinIcon /> MUMBAI TRAFFIC NETWORK
            </div>
          </div>
        </div>
      </section>

      {/* MAP CONTROLS — a separate layout element below the map, never overlapping it */}
      <div className="bg-slate-900 border border-slate-700/80 rounded-xl p-3 shadow-lg shrink-0 flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-2 lg:gap-3">
          <div
            className="flex items-center gap-1 bg-slate-950 border border-slate-800 rounded-lg p-1"
            role="group"
            aria-label="Map type"
          >
            {MAP_TYPE_OPTIONS.map((opt) => (
              <button
                key={opt.id}
                type="button"
                onClick={() => setMapTypeId(opt.id)}
                title={`Map type: ${opt.label}`}
                aria-pressed={mapTypeId === opt.id}
                className={`px-2.5 py-1.5 rounded text-[10px] font-bold uppercase tracking-wide transition-colors ${mapTypeId === opt.id ? "bg-blue-600 text-white" : "text-slate-400 hover:text-slate-200"}`}
              >
                {opt.label}
              </button>
            ))}
          </div>

          <div
            className="flex items-center gap-1 bg-slate-950 border border-slate-800 rounded-lg p-1"
            role="group"
            aria-label="Traffic view mode"
          >
            {["LIVE", "PREDICTED", "HISTORICAL"].map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => setTrafficViewMode(mode)}
                aria-pressed={trafficViewMode === mode}
                className={`px-2.5 py-1.5 rounded text-[10px] font-bold uppercase tracking-wide transition-colors ${
                  trafficViewMode === mode
                    ? "bg-blue-600 text-white"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {mode}
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={() => setTrafficOn((v) => !v)}
            title="Toggle the live Google traffic layer when LIVE mode is selected"
            aria-pressed={trafficOn}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-wide border transition-colors ${trafficOn ? "bg-blue-900/30 border-blue-500/50 text-blue-300" : "bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200"}`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${trafficOn ? "bg-blue-400 animate-pulse" : "bg-slate-600"}`}
            ></span>
            Traffic {trafficOn ? "On" : "Off"}
          </button>

          <div className="flex items-center gap-1.5">
            <label
              htmlFor="traffic-forecast-select"
              className="text-[10px] font-bold uppercase tracking-widest text-slate-500"
            >
              Peak Forecast
            </label>
            <select
              id="traffic-forecast-select"
              value={forecastMode}
              onChange={(event) => setForecastMode(event.target.value)}
              title="Google-predicted traffic-aware travel time for this corridor at a given time of day"
              className="bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-[10px] font-bold uppercase tracking-wide text-slate-300 outline-none focus:border-blue-500"
            >
              {FORECAST_OPTIONS.map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
            {forecastMode !== "Now" && (
              <span className="text-[9px] bg-blue-900/30 text-blue-300 border border-blue-500/40 px-2 py-0.5 rounded font-mono uppercase">
                Predicted · Google Traffic Model
              </span>
            )}
          </div>

          {trafficViewMode === "HISTORICAL" && (
            <div className="flex items-center gap-2 flex-wrap">
              <label className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
                History
              </label>
              <input
                type="date"
                value={historyDate}
                onChange={(event) => setHistoryDate(event.target.value)}
                className="bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-[10px] font-bold text-slate-300 outline-none focus:border-blue-500"
              />
              <input
                type="time"
                value={historyTime}
                onChange={(event) => setHistoryTime(event.target.value)}
                className="bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-[10px] font-bold text-slate-300 outline-none focus:border-blue-500"
              />
              <span className="text-[9px] bg-amber-900/30 text-amber-300 border border-amber-500/40 px-2 py-0.5 rounded font-mono uppercase">
                Archived Google observations
              </span>
            </div>
          )}

          <div className="flex-1 hidden lg:block" />

          <div className="flex items-center gap-2 flex-wrap">
            <button
              type="button"
              onClick={handleCenterOnMumbai}
              title="Center on Mumbai"
              className="px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-wide bg-slate-950 border border-slate-800 text-slate-300 hover:border-slate-600 transition-colors"
            >
              Mumbai
            </button>
            <button
              type="button"
              onClick={handleFitTrafficAreas}
              title="Fit all traffic areas in view"
              className="px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-wide bg-slate-950 border border-slate-800 text-slate-300 hover:border-slate-600 transition-colors"
            >
              Fit Traffic Areas
            </button>
            <button
              type="button"
              onClick={handleFocusSelected}
              disabled={!activeNode}
              title="Focus the currently selected location"
              className="px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-wide bg-slate-950 border border-slate-800 text-slate-300 hover:border-slate-600 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Focus Location
            </button>
          </div>
        </div>

        {trafficViewMode === "PREDICTED" && (
          <div className="pt-2 border-t border-slate-800 text-[11px] flex flex-wrap items-center gap-x-4 gap-y-1">
            <span className="text-blue-400 font-bold uppercase tracking-widest text-[10px]">
              PREDICTED TRAFFIC ON MAP
            </span>
            <span className="text-slate-400">
              Google traffic-aware route segments:
              <span className="text-emerald-400 ml-1">NORMAL</span>
              <span className="text-amber-400 ml-2">SLOW</span>
              <span className="text-red-400 ml-2">TRAFFIC JAM</span>
            </span>
            {forecastLoading && (
              <span className="text-slate-500 italic">
                Querying Google traffic model...
              </span>
            )}
            {forecastError && (
              <span className="text-red-400">{forecastError}</span>
            )}
          </div>
        )}

        {trafficViewMode === "HISTORICAL" && (
          <div className="pt-2 border-t border-slate-800 text-[11px] flex flex-wrap items-center gap-x-4 gap-y-1">
            <span className="text-amber-400 font-bold uppercase tracking-widest text-[10px]">
              HISTORICAL TRAFFIC
            </span>
            {historyLoading ? (
              <span className="text-slate-500 italic">
                Loading archived observations...
              </span>
            ) : historyError ? (
              <span className="text-amber-300">{historyError}</span>
            ) : (
              <span className="text-slate-300">
                {historicalTraffic.length} archived corridor observation
                {historicalTraffic.length === 1 ? "" : "s"} · {historyDate} {historyTime} IST
              </span>
            )}
          </div>
        )}

        {/* Peak-forecast readout — a real Google Routes API prediction for the
            current origin → destination corridor, never a fabricated number. */}
        {forecastMode !== "Now" && (
          <div className="pt-2 border-t border-slate-800 text-[11px] flex flex-wrap items-center gap-x-4 gap-y-1">
            <span className="text-slate-500 font-bold uppercase tracking-widest text-[10px]">
              {origin} → {destination} @ {forecastMode}
              {forecastResult?.departureAt
                ? ` (${forecastResult.departureAt.toLocaleString("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", day: "2-digit", month: "short" })} IST)`
                : ""}
              :
            </span>
            {forecastLoading ? (
              <span className="text-slate-500 italic">
                Querying Google traffic model...
              </span>
            ) : forecastError ? (
              <span className="text-red-400">{forecastError}</span>
            ) : forecastResult ? (
              <>
                <span className="text-slate-200 font-mono font-bold">
                  {forecastResult.route.duration}
                </span>
                <span className="text-slate-400">
                  delay{" "}
                  <span className="text-orange-400 font-bold">
                    +{forecastResult.route.delayMinutes} min
                  </span>{" "}
                  vs free-flow
                </span>
                {forecastResult.route.averageSpeedKmh != null && (
                  <span className="text-slate-400">
                    avg{" "}
                    <span className="text-slate-200 font-bold">
                      {forecastResult.route.averageSpeedKmh} km/h
                    </span>
                  </span>
                )}
                <span
                  className={`font-bold uppercase ${severityTextColor(severityFromDelayRatio(forecastResult.route.delayMinutes, forecastResult.route.staticDurationSeconds))}`}
                >
                  {severityFromDelayRatio(
                    forecastResult.route.delayMinutes,
                    forecastResult.route.staticDurationSeconds,
                  )}
                </span>
              </>
            ) : (
              <span className="text-slate-500 italic">
                No prediction available yet.
              </span>
            )}
          </div>
        )}
      </div>

      {/* TRAFFIC AGENT LOG — fixed height, internal scroll only, never moves the page */}
      {selectedScenarioId && (
        <div className="relative bg-black/90 backdrop-blur border border-slate-700/80 rounded-xl p-3 shadow-lg h-36 flex flex-col overflow-hidden animate-fade-in shrink-0">
          <div className="flex items-center gap-2 text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2 border-b border-slate-800 pb-1.5 shrink-0">
            <TerminalIcon /> TRAFFIC AGENT LOG
          </div>
          <div
            ref={logScrollRef}
            onScroll={handleLogScroll}
            className="flex-1 overflow-y-auto custom-scrollbar font-mono text-[11px] leading-relaxed space-y-1"
          >
            {terminalLogs.length === 0 ? (
              <span className="text-slate-600">
                Awaiting execution sequence...
              </span>
            ) : (
              terminalLogs.map((log, i) => (
                <div key={i} className="text-green-400">
                  {log}
                </div>
              ))
            )}
          </div>
          {hasNewLogActivity && !autoScrollLog && (
            <button
              type="button"
              onClick={scrollLogToBottom}
              className="absolute bottom-2 right-3 text-[9px] font-bold uppercase tracking-widest bg-blue-600 hover:bg-blue-500 text-white px-2 py-1 rounded-full shadow-lg animate-fade-in"
            >
              New activity ↓
            </button>
          )}
        </div>
      )}

      {/* DYNAMIC VIEW LAYER: SCENARIO REPLAY vs. MANUAL EXPLORATION */}
      {isReplayView ? (
        /* --------------------------------------------------------
           SCENARIO REPLAY: all cards shown together, smooth fade-in
           (Route Intelligence removed; stage badges removed)
           -------------------------------------------------------- */
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-fade-in">
          <div className="bg-slate-900/80 p-5 rounded-xl border border-slate-700/50 shadow-lg">
            <h3 className="text-red-400 font-bold uppercase tracking-widest text-sm mb-4">
              Traffic Condition Detected
            </h3>
            <div className="text-xl font-bold text-slate-100 mb-3">
              {selectedScenario?.location || "Dadar Junction"}
            </div>
            <div className="space-y-2 mb-4">
              <div className="flex justify-between border-b border-slate-800 pb-1">
                <span className="text-slate-500 text-xs uppercase font-bold">
                  Traffic State
                </span>
                <span
                  className={`${scenarioColor(selectedScenario?.severity)} font-bold text-sm`}
                >
                  {selectedScenario?.severity}
                </span>
              </div>
              <div className="flex justify-between border-b border-slate-800 pb-1">
                <span className="text-slate-500 text-xs uppercase font-bold">
                  Density
                </span>
                <span className="text-orange-400 font-bold text-sm">HIGH</span>
              </div>
              <div className="flex justify-between border-b border-slate-800 pb-1">
                <span className="text-slate-500 text-xs uppercase font-bold">
                  Flow
                </span>
                <span className="text-red-400 font-bold text-sm">
                  DETERIORATING
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2 text-xs font-mono text-blue-400 font-bold">
              <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>{" "}
              RECEIVING TELEMETRY
            </div>
          </div>

          <div className="bg-slate-900/80 p-5 rounded-xl border border-slate-700/50 shadow-lg">
            <h3 className="text-blue-400 font-bold uppercase tracking-widest text-sm mb-4">
              Traffic Agent Analysis
            </h3>
            <ul className="space-y-3 text-sm text-slate-300">
              <li className="flex gap-2 items-start">
                <CheckIcon />{" "}
                <span className="pt-0.5">
                  {scenarioContent?.analysis || "Assessing affected corridor."}
                </span>
              </li>
              <li className="flex gap-2 items-start">
                <CheckIcon />{" "}
                <span className="pt-0.5">
                  Evaluating adjacent nodes for secondary congestion.
                </span>
              </li>
              <li className="flex gap-2 items-start">
                <CheckIcon />{" "}
                <span className="pt-0.5">
                  Calculating necessary dynamic interventions.
                </span>
              </li>
            </ul>
          </div>

          <div className="bg-slate-900/80 p-5 rounded-xl border border-slate-700/50 shadow-lg">
            <h3 className="text-indigo-400 font-bold uppercase tracking-widest text-sm mb-4">
              AI Traffic Forecast
            </h3>
            <div className="mb-4 inline-block text-[9px] font-mono bg-indigo-900/30 text-indigo-300 border border-indigo-500/50 px-2 py-0.5 rounded">
              DEMONSTRATION / SIMULATED PREDICTION
            </div>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="bg-slate-950 p-3 rounded border border-slate-800">
                <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">
                  Current
                </div>
                <div className="text-sm font-bold text-orange-400">
                  {selectedScenario?.severity || "HIGH"}
                </div>
              </div>
              <div className="bg-slate-950 p-3 rounded border border-slate-800">
                <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">
                  Expected (+15 min)
                </div>
                <div className="text-sm font-bold text-red-500">SEVERE</div>
              </div>
            </div>
            <div className="bg-slate-950 p-3 rounded border border-slate-800">
              <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                Prediction Reason:
              </span>
              <p className="text-xs text-slate-300 mt-1">
                {scenarioContent?.prediction ||
                  "Sustained volume indicating gridlock."}
              </p>
            </div>
          </div>

          <div className="bg-slate-900/80 p-5 rounded-xl border border-slate-700/50 shadow-lg">
            <h3 className="text-emerald-400 font-bold uppercase tracking-widest text-sm mb-4">
              Signal Optimization
            </h3>
            <div className="mb-4 inline-block text-[9px] font-mono bg-emerald-900/30 text-emerald-400 border border-emerald-500/50 px-2 py-0.5 rounded">
              SIMULATED OPTIMIZATION
            </div>

            <div className="grid grid-cols-2 gap-6 mb-4">
              <div>
                <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-2">
                  Current Signal
                </div>
                <div className="space-y-1.5">
                  <div className="flex items-center text-xs">
                    <span className="w-10 font-bold text-slate-400">GREEN</span>
                    <div className="flex-1 bg-slate-800 h-2 mx-2 rounded overflow-hidden">
                      <div className="bg-emerald-600 h-full w-[42%]"></div>
                    </div>
                    <span className="w-6 font-mono text-slate-400 text-right">
                      {scenarioContent?.signalDiff?.currentGreen}s
                    </span>
                  </div>
                  <div className="flex items-center text-xs">
                    <span className="w-10 font-bold text-slate-400">RED</span>
                    <div className="flex-1 bg-slate-800 h-2 mx-2 rounded overflow-hidden">
                      <div className="bg-red-600 h-full w-[58%]"></div>
                    </div>
                    <span className="w-6 font-mono text-slate-400 text-right">
                      {scenarioContent?.signalDiff?.currentRed}s
                    </span>
                  </div>
                </div>
              </div>
              <div>
                <div className="text-[10px] text-blue-400 font-bold uppercase tracking-widest mb-2 flex items-center gap-1">
                  <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse"></span>{" "}
                  AI Optimized
                </div>
                <div className="space-y-1.5">
                  <div className="flex items-center text-xs">
                    <span className="w-10 font-bold text-slate-400">GREEN</span>
                    <div className="flex-1 bg-slate-800 h-2 mx-2 rounded overflow-hidden">
                      <div className="bg-emerald-400 h-full w-[54%] transition-all duration-1000 shadow-[0_0_8px_rgba(52,211,153,0.5)]"></div>
                    </div>
                    <span className="w-6 font-mono text-emerald-400 text-right">
                      {scenarioContent?.signalDiff?.optGreen}s
                    </span>
                  </div>
                  <div className="flex items-center text-xs">
                    <span className="w-10 font-bold text-slate-400">RED</span>
                    <div className="flex-1 bg-slate-800 h-2 mx-2 rounded overflow-hidden">
                      <div className="bg-red-400 h-full w-[46%] transition-all duration-1000 shadow-[0_0_8px_rgba(248,113,113,0.5)]"></div>
                    </div>
                    <span className="w-6 font-mono text-red-400 text-right">
                      {scenarioContent?.signalDiff?.optRed}s
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-slate-950 p-3 rounded border border-slate-800 text-xs flex justify-between items-center">
              <span className="text-emerald-400 font-bold uppercase tracking-widest text-[10px]">
                Action:
              </span>
              <span className="font-mono text-slate-300 font-bold bg-slate-800 px-2 py-1 rounded truncate">
                GREEN {scenarioContent?.signalDiff?.greenDiff} | RED{" "}
                {scenarioContent?.signalDiff?.redDiff}
              </span>
            </div>
          </div>

          <div className="bg-slate-900/80 border border-blue-500/50 shadow-[0_0_15px_rgba(59,130,246,0.15)] rounded-xl p-5">
            <div className="flex justify-between items-center mb-4 border-b border-slate-700/50 pb-2">
              <h3 className="font-bold text-sm uppercase tracking-widest text-slate-200 flex items-center gap-2">
                <BrainIcon /> TRAFFIC AGENT DECISION
              </h3>
            </div>

            <div className="text-lg font-bold text-slate-100">
              {selectedScenario?.location || "Dadar Junction"}
            </div>
            <div
              className={`text-sm font-bold uppercase mb-4 ${scenarioColor(selectedScenario?.severity)}`}
            >
              {selectedScenario?.severity} TRAFFIC
            </div>

            <div className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-1">
              ACTION
            </div>
            <div className="text-sm font-bold text-blue-400 uppercase tracking-wider mb-4 bg-slate-950 p-2 rounded border border-slate-800">
              {scenarioContent?.decision || "ANALYZING"}
            </div>

            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="bg-slate-950 p-3 rounded border border-slate-800">
                <div className="text-[10px] font-bold text-slate-500 mb-1">
                  GREEN
                </div>
                <div className="text-xs font-mono font-bold text-slate-300">
                  {scenarioContent?.signalDiff?.currentGreen}s{" "}
                  <span className="text-emerald-400">
                    → {scenarioContent?.signalDiff?.optGreen}s
                  </span>
                </div>
              </div>
              <div className="bg-slate-950 p-3 rounded border border-slate-800">
                <div className="text-[10px] font-bold text-slate-500 mb-1">
                  RED
                </div>
                <div className="text-xs font-mono font-bold text-slate-300">
                  {scenarioContent?.signalDiff?.currentRed}s{" "}
                  <span className="text-emerald-400">
                    → {scenarioContent?.signalDiff?.optRed}s
                  </span>
                </div>
              </div>
            </div>

            <div className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-1">
              WHY
            </div>
            <div className="text-xs text-slate-300">
              {scenarioContent?.analysis || "Pending analysis."}
            </div>
          </div>

          <div className="bg-slate-900/80 border border-emerald-500/50 shadow-[0_0_15px_rgba(52,211,153,0.15)] rounded-xl p-5">
            <div className="flex justify-between items-center mb-4 border-b border-slate-700/50 pb-2">
              <h3 className="font-bold text-sm uppercase tracking-widest text-slate-200 flex items-center gap-2">
                <SignalIcon /> SIMULATED IMPACT
              </h3>
            </div>

            <div className="space-y-4">
              <div>
                <div className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-1">
                  Traffic State
                </div>
                <div className="flex items-center gap-2 text-xs font-bold bg-slate-950 p-2 rounded border border-slate-800">
                  <span className={scenarioColor(selectedScenario?.severity)}>
                    {selectedScenario?.severity || "SEVERE"}
                  </span>
                  <span className="text-slate-500">→</span>
                  <span className="text-emerald-400">IMPROVING</span>
                </div>
              </div>
              <div>
                <div className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-1">
                  Route
                </div>
                <div className="flex items-center gap-2 text-xs font-bold bg-slate-950 p-2 rounded border border-slate-800">
                  <span className="text-slate-400">PRIMARY</span>
                  <span className="text-slate-500">→</span>
                  <span
                    className="text-indigo-400 uppercase truncate"
                    title={scenarioContent?.outcomeRoute}
                  >
                    {scenarioContent?.outcomeRoute || "ALTERNATIVE"}
                  </span>
                </div>
              </div>
              <div>
                <div className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-1">
                  Signal
                </div>
                <div className="flex items-center gap-2 text-xs font-bold bg-slate-950 p-2 rounded border border-slate-800">
                  <span className="text-slate-400">
                    {scenarioContent?.signalDiff?.currentGreen || 30}s
                  </span>
                  <span className="text-slate-500">→</span>
                  <span className="text-emerald-400">
                    {scenarioContent?.outcomeSignal || "GREEN"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* --------------------------------------------------------
           STANDARD DASHBOARD: MANUAL EXPLORATION (no Route Intelligence)
           -------------------------------------------------------- */
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 shrink-0 mt-6">
            {/* Traffic Agent */}
            <div className="bg-slate-900/80 border border-slate-700/50 rounded-xl shadow-lg flex flex-col overflow-hidden">
              <div className="p-4 border-b border-slate-700/50 bg-slate-800/30 flex justify-between items-center">
                <h3 className="font-bold text-sm uppercase tracking-widest text-slate-200 flex items-center gap-2">
                  <BrainIcon /> Traffic Agent
                </h3>
                <span className="text-[9px] bg-slate-950 px-2 py-1 rounded text-slate-400 font-mono border border-slate-800 uppercase">
                  {isDefaultFocus ? telemetrySourceLabel : demoTraffic.source}
                </span>
              </div>
              <div className="p-5 flex-1">
                {!activeNode ? (
                  <div className="text-slate-500 text-sm italic">
                    Loading traffic telemetry...
                  </div>
                ) : (
                  <div className="space-y-4 animate-fade-in">
                    {isDefaultFocus && (
                      <div className="inline-block text-[9px] font-mono bg-slate-950 text-slate-400 border border-slate-700 px-2 py-0.5 rounded uppercase">
                        Default Traffic Advisory
                      </div>
                    )}

                    <div className="text-sm font-bold text-slate-400 uppercase tracking-widest">
                      {isDefaultFocus ? "Monitoring" : "Selected"}:{" "}
                      <span className="text-slate-100 ml-1">
                        {activeNode.name}
                      </span>
                    </div>

                    <div className="flex items-center gap-2 text-blue-400 font-mono text-xs font-bold uppercase tracking-widest my-4">
                      <span
                        className={`w-2 h-2 rounded-full ${isDefaultFocus ? "bg-slate-500" : "bg-blue-500 animate-pulse"}`}
                      ></span>
                      {isDefaultFocus
                        ? "No active simulation decision"
                        : agentStatus}
                    </div>

                    <div>
                      <span className="text-xs text-slate-500 font-bold uppercase tracking-widest mb-1 block">
                        Current:
                      </span>
                      <span
                        className={`text-lg font-bold uppercase ${activeNode.color || severityTextColor(activeNode.severity)}`}
                      >
                        {activeNode.severity}
                      </span>
                    </div>

                    <div>
                      <span className="text-xs text-slate-500 font-bold uppercase tracking-widest mb-1 block">
                        Prediction:
                      </span>
                      <span className="text-sm font-bold uppercase text-blue-300">
                        {agentDecision?.prediction
                          ? `${agentDecision.prediction.predictedState} in ${agentDecision.prediction.horizonMinutes} min`
                          : `${nextSeverity(activeNode.severity)} in 15 min (demo forecast)`}
                      </span>
                    </div>

                    {isDefaultFocus && (
                      <div className="text-xs text-slate-500 pt-2 border-t border-slate-800">
                        Monitor {activeNode.name} corridor — no active
                        simulation decision. Click a location for live
                        agent analysis.
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Signal Optimization */}
            <div className="bg-slate-900/80 border border-slate-700/50 rounded-xl shadow-lg flex flex-col overflow-hidden">
              <div className="p-4 border-b border-slate-700/50 bg-slate-800/30 flex justify-between items-center">
                <h3 className="font-bold text-sm uppercase tracking-widest text-slate-200 flex items-center gap-2">
                  <SignalIcon /> Signal Optimization
                </h3>
                {isDefaultSignal && (
                  <span className="text-[9px] bg-slate-950 px-2 py-1 rounded text-slate-400 font-mono border border-slate-800 uppercase">
                    Demo Telemetry
                  </span>
                )}
              </div>
              <div className="p-5 flex-1 overflow-y-auto custom-scrollbar">
                {!activeNode ? (
                  <div className="text-slate-500 text-sm italic">
                    Loading signal telemetry...
                  </div>
                ) : (
                  <div className="space-y-4 animate-fade-in">
                    <div className="grid grid-cols-2 gap-4 border-b border-slate-700/50 pb-4">
                      <div>
                        <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-3">
                          Current
                        </div>
                        {renderSignalBlock({
                          currentGreen: signalSource.currentGreen,
                          currentRed: signalSource.currentRed,
                        })}
                      </div>
                      <div>
                        <div className="text-[10px] text-blue-400 font-bold uppercase tracking-widest mb-3 flex items-center gap-1">
                          <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse"></span>{" "}
                          AI Optimized
                        </div>
                        <div className="flex flex-col gap-2 mb-3">
                          <div className="flex justify-between items-center text-xs">
                            <span className="font-bold text-slate-400 w-16">
                              GREEN
                            </span>
                            <div className="flex-1 mx-2 flex gap-4 items-center">
                              <div className="h-2 flex-1 rounded bg-slate-800 overflow-hidden shadow-[0_0_8px_rgba(59,130,246,0.3)]">
                                <div
                                  className="h-full bg-emerald-400"
                                  style={{
                                    width: `${(signalSource.optimizedGreen / (signalSource.optimizedGreen + signalSource.optimizedRed)) * 100}%`,
                                  }}
                                ></div>
                              </div>
                              <span className="font-mono text-emerald-400 w-8">
                                {signalSource.optimizedGreen}s
                              </span>
                            </div>
                          </div>
                          <div className="flex justify-between items-center text-xs">
                            <span className="font-bold text-slate-400 w-16">
                              RED
                            </span>
                            <div className="flex-1 mx-2 flex gap-4 items-center">
                              <div className="h-2 flex-1 rounded bg-slate-800 overflow-hidden shadow-[0_0_8px_rgba(59,130,246,0.3)]">
                                <div
                                  className="h-full bg-red-400"
                                  style={{
                                    width: `${(signalSource.optimizedRed / (signalSource.optimizedGreen + signalSource.optimizedRed)) * 100}%`,
                                  }}
                                ></div>
                              </div>
                              <span className="font-mono text-red-400 w-8">
                                {signalSource.optimizedRed}s
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div>
                      <div className="text-sm font-bold text-slate-200 mb-1">
                        Recommendation:{" "}
                        <span className="text-emerald-400 ml-1">
                          {signalSource.greenAdjustment
                            ? `Simulated GREEN +${signalSource.greenAdjustment}s`
                            : isDefaultSignal
                              ? "Maintain current timing (demo baseline)"
                              : "Maintain current timing"}
                        </span>
                      </div>
                      <div className="text-xs text-slate-400">
                        Reason:{" "}
                        {agentDecision?.prediction?.reason ||
                          (isDefaultSignal
                            ? "Default signal-timing baseline for this corridor; no live agent evaluation has run yet."
                            : "Demand and safe signal timing limits were evaluated.")}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* AI PREDICTIONS & AGENT PERFORMANCE */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 shrink-0 mt-6">
            <div className="bg-slate-900/80 border border-slate-700/50 rounded-xl shadow-lg flex flex-col overflow-hidden h-64">
              <div className="p-4 border-b border-slate-700/50 bg-slate-800/30 flex justify-between items-center">
                <h3 className="font-bold text-sm uppercase tracking-widest text-slate-200">
                  AI Predictions
                </h3>
                {!agentDecision?.prediction && (
                  <span className="text-[9px] bg-slate-950 px-2 py-1 rounded text-slate-400 font-mono border border-slate-800 uppercase">
                    Demo Forecast
                  </span>
                )}
              </div>
              <div className="p-5 flex flex-col justify-center gap-4 flex-1">
                <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                  <span className="text-sm font-bold text-slate-400">
                    Congestion
                  </span>
                  <span className="text-sm font-bold text-blue-300">
                    {agentDecision?.prediction
                      ? `${agentDecision.prediction.predictedState} · ${agentDecision.prediction.horizonMinutes} min`
                      : activeNode
                        ? `${nextSeverity(activeNode.severity)} · 15 min`
                        : "Awaiting telemetry"}
                  </span>
                </div>
                <div className="flex justify-between items-center border-b border-slate-800 pb-3">
                  <span className="text-sm font-bold text-slate-400">
                    Density
                  </span>
                  <span className="text-sm font-bold text-blue-300">
                    {agentDecision?.densityPrediction
                      ? `${agentDecision.densityPrediction.predictedState} · ${agentDecision.densityPrediction.horizonMinutes} min`
                      : activeNode
                        ? `${nextSeverity(activeNode.severity)} · 20 min`
                        : "Awaiting telemetry"}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm font-bold text-slate-400">
                    Best Route
                  </span>
                  <span
                    className={`text-sm font-bold ${bestRoute !== "Unavailable" ? "text-blue-400" : "text-slate-500 italic"}`}
                  >
                    {bestRoute}
                  </span>
                </div>
                {!agentDecision?.prediction && activeNode && (
                  <div className="text-[11px] text-slate-500 pt-1 border-t border-slate-800">
                    Peak-period traffic building across adjacent corridors
                    with reduced average speed near {activeNode.name}.
                  </div>
                )}
              </div>
            </div>

            <div className="bg-slate-900/80 border border-slate-700/50 rounded-xl shadow-lg flex flex-col overflow-hidden h-64">
              <div className="p-4 border-b border-slate-700/50 bg-slate-800/30">
                <h3 className="font-bold text-sm uppercase tracking-widest text-slate-200">
                  Traffic Agent Performance
                </h3>
              </div>
              <div className="p-5 grid grid-cols-2 gap-4 flex-1">
                <div className="flex justify-between items-center bg-slate-950 p-3 rounded-lg border border-slate-800">
                  <span className="text-xs font-bold text-slate-400">
                    Signal Optimizations
                  </span>
                  <span className="text-lg font-mono font-bold text-slate-300">
                    {agentMetrics.signalOptimizations}
                  </span>
                </div>
                <div className="flex justify-between items-center bg-slate-950 p-3 rounded-lg border border-slate-800">
                  <span className="text-xs font-bold text-slate-400">
                    Successful Recs
                  </span>
                  <span className="text-lg font-mono font-bold text-slate-300">
                    {agentMetrics.successfulRecommendations}
                  </span>
                </div>
                <div className="flex justify-between items-center bg-slate-950 p-3 rounded-lg border border-slate-800">
                  <span className="text-xs font-bold text-slate-400">
                    Route Recs
                  </span>
                  <span className="text-lg font-mono font-bold text-slate-300">
                    {agentMetrics.routeRecommendations}
                  </span>
                </div>
                <div className="flex justify-between items-center bg-slate-950 p-3 rounded-lg border border-slate-800">
                  <span className="text-xs font-bold text-slate-400">
                    Prediction Alerts
                  </span>
                  <span className="text-lg font-mono font-bold text-slate-300">
                    {agentMetrics.predictionAlerts}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      <style
        dangerouslySetInnerHTML={{
          __html: `
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        .animate-fade-in { animation: fadeIn 0.4s ease-out forwards; }
      `,
        }}
      />
    </div>
  );
}