import { create } from 'zustand';

export const useGlobalStore = create((set) => ({
  simulationTime: '00:00',
  systemStatus: 'Nominal',
  activeAlerts: [],
  aiRecommendations: [],
  
  // Actions to be called by Socket.IO later
  setSimulationTime: (time) => set({ simulationTime: time }),
  addAlert: (alert) => set((state) => ({ 
    activeAlerts: [alert, ...state.activeAlerts].slice(0, 10) 
  })),
  addRecommendation: (rec) => set((state) => ({ 
    aiRecommendations: [rec, ...state.aiRecommendations].slice(0, 5) 
  })),
}));