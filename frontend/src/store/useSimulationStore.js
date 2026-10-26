import { create } from 'zustand';

export const useSimulationStore = create((set) => ({
  // Global Simulation State
  isSimulationActive: false,
  engineTick: 0,
  scenarioName: 'Idle',
  
  // Incident State
  activeIncident: null, // e.g., { id, type, location, severity, coordinates }
  
  // Department Live Metrics
  departments: {
    traffic: { status: 'Nominal', alertLevel: 'Low', data: {} },
    hospital: { status: 'Standby', alertLevel: 'Low', data: {} },
    police: { status: 'Patrolling', alertLevel: 'Low', data: {} },
    fire: { status: 'Standby', alertLevel: 'Low', data: {} },
    utility: { status: 'Nominal', alertLevel: 'Low', data: {} },
    citizen: { status: 'Normal', alertLevel: 'Low', data: {} }
  },

  // Decision Intelligence (Command Center Agent)
  actionPlan: [],
  consensusScore: 100,

  // LLM Logs
  assistantLogs: [],

  // --- ACTIONS ---

  setSystemTick: (tick, activeScenario) => set({ 
    engineTick: tick, 
    scenarioName: activeScenario,
    isSimulationActive: true 
  }),

  triggerIncident: (incidentData) => set({
    activeIncident: incidentData
  }),

  updateDepartmentMetrics: (deptName, payload) => set((state) => ({
    departments: {
      ...state.departments,
      [deptName]: { ...state.departments[deptName], ...payload }
    }
  })),

  updateActionPlan: (plan, score) => set({
    actionPlan: plan,
    consensusScore: score
  }),

  addAssistantLog: (log) => set((state) => ({
    assistantLogs: [...state.assistantLogs, log]
  })),

  resetSimulation: () => set({
    isSimulationActive: false,
    engineTick: 0,
    activeIncident: null,
    actionPlan: [],
    assistantLogs: []
  })
}));