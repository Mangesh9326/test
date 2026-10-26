import { create } from 'zustand';

export const SCENARIO_STEPS = ["DETECT", "ANALYZE", "PREDICT", "DECIDE", "OPTIMIZE", "ROUTE", "OUTCOME"];

let stepTimer = null;

// ---------------------------------------------------------------------
// CROSS-DEVICE SYNC
// This store only lives in the current browser tab's memory, so a Live
// Traffic display on another device can't see it directly. Every scenario
// lifecycle action below also POSTs the resulting playback state to the
// backend, which any LiveTraffic page (on this device or another) polls
// via GET /api/traffic/playback-state and mirrors automatically.
// ---------------------------------------------------------------------
const PLAYBACK_STATE_URL = "http://localhost:5000/api/traffic/playback-state";

// Dedup guard: skip POSTing the exact same {scenarioId, isRunning, isPaused}
// twice in a row (e.g. React StrictMode double-invokes, or a handler firing
// more than once for the same transition).
let lastPostedSignature = null;

const postPlaybackState = async ({ scenarioId, isRunning, isPaused }) => {
  const normalized = { scenarioId: scenarioId || null, isRunning: !!isRunning, isPaused: !!isPaused };
  const signature = `${normalized.scenarioId}|${normalized.isRunning}|${normalized.isPaused}`;
  if (signature === lastPostedSignature) return;
  lastPostedSignature = signature;

  try {
    await fetch(PLAYBACK_STATE_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(normalized),
    });
  } catch (err) {
    console.error("Failed to sync playback state to server:", err);
    // Allow a retry on the next call since this attempt failed.
    lastPostedSignature = null;
  }
};

export const useTrafficControlStore = create((set, get) => ({
  // data
  scenarios: [],
  scenariosLoading: false,
  scenariosError: null,

  // selection / playback
  selectedScenarioId: "",
  selectedScenario: null,
  scenarioStep: -1,
  isRunning: false,
  isPaused: false,
  agentStatus: "IDLE",
  agentDecision: null,

  // ---- data fetching ----
  fetchScenarios: async () => {
    set({ scenariosLoading: true, scenariosError: null });
    try {
      const res = await fetch("http://localhost:5000/api/traffic/scenarios");
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to load scenarios");
      set({ scenarios: data.scenarios || [], scenariosLoading: false });
    } catch (err) {
      console.error("Failed to fetch scenarios:", err);
      set({ scenariosError: err.message, scenariosLoading: false });
    }
  },

  // ---- scenario lifecycle ----
  prepareScenario: async (scenarioId) => {
    clearTimeout(stepTimer);
    set({
      scenarioStep: -1,
      isRunning: false,
      isPaused: false,
      agentDecision: null,
      agentStatus: "SCENARIO READY",
    });

    if (!scenarioId) {
      set({ selectedScenarioId: "", selectedScenario: null, agentStatus: "IDLE" });
      postPlaybackState({ scenarioId: null, isRunning: false, isPaused: false });
      return;
    }

    try {
      const res = await fetch(`http://localhost:5000/api/traffic/scenarios/${scenarioId}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to prepare scenario");
      set({
        selectedScenarioId: String(data.scenario.id),
        selectedScenario: data.scenario,
      });
      postPlaybackState({ scenarioId: data.scenario.id, isRunning: false, isPaused: false });
    } catch (err) {
      console.error("Scenario preparation failed:", err);
      set({ agentStatus: "SCENARIO UNAVAILABLE" });
    }
  },

  runScenario: () => {
    const { selectedScenarioId, isRunning, scenarioStep } = get();
    if (!selectedScenarioId || isRunning) return;

    let step = scenarioStep;
    if (step >= SCENARIO_STEPS.length - 1) step = -1;

    set({ isRunning: true, isPaused: false });
    postPlaybackState({ scenarioId: selectedScenarioId, isRunning: true, isPaused: false });

    const advance = async () => {
      step = step + 1;
      set({ scenarioStep: step });

      if (step < SCENARIO_STEPS.length - 1) {
        stepTimer = setTimeout(advance, 1500);
        return;
      }

      try {
        const { selectedScenarioId: id } = get();
        const res = await fetch(`http://localhost:5000/api/traffic/scenarios/${id}/replay`, { method: "POST" });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Replay failed");
        set({ agentDecision: data.decision?.metadata || null, agentStatus: "SCENARIO COMPLETE" });
      } catch (err) {
        console.error("Scenario replay failed:", err);
        set({ agentStatus: "REPLAY UNAVAILABLE" });
      } finally {
        const { selectedScenarioId: id } = get();
        set({ isRunning: false });
        postPlaybackState({ scenarioId: id, isRunning: false, isPaused: false });
      }
    };
    advance();
  },

  pauseScenario: () => {
    clearTimeout(stepTimer);
    set({ isRunning: false, isPaused: true });
    postPlaybackState({ scenarioId: get().selectedScenarioId, isRunning: false, isPaused: true });
  },

  resetScenario: () => {
    const { selectedScenarioId, prepareScenario } = get();
    if (selectedScenarioId) prepareScenario(selectedScenarioId);
  },
}));