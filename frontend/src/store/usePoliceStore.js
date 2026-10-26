import { create } from 'zustand';
import socketService from '../services/socketService';

export const usePoliceStore = create((set, get) => ({
  overviewMetrics: { activeIncidents: 'N/A', criticalIncidents: 'N/A', unitsAvailable: 'N/A', unitsDeployed: 'N/A' },
  activeIncidents: [],
  crimes: [],
  agentRecommendation: null,
  coordinationEvents: [],
  isLoading: true,
  isSimulation: true,
  
  fetchInitialData: async () => {
    set({ isLoading: true });
    try {
      const [overviewRes, incidentsRes, crimesRes, agentRes] = await Promise.all([
        fetch('/api/police/overview').then(r => r.json()),
        fetch('/api/police/incidents').then(r => r.json()),
        fetch('/api/police/crimes').then(r => r.json()),
        fetch('/api/police/agent/recommendation').then(r => r.json())
      ]);
      
      set({ 
        overviewMetrics: overviewRes, 
        activeIncidents: incidentsRes,
        crimes: crimesRes,
        agentRecommendation: agentRes,
        isLoading: false 
      });
    } catch (error) {
      console.error("[Police Store] Initialization error:", error);
      set({ isLoading: false });
    }
  },

  subscribeToSockets: () => {
    const socket = socketService.socket || (typeof socketService.getSocket === 'function' ? socketService.getSocket() : socketService);
    if (!socket || typeof socket.on !== 'function') return;

    socket.on('INCIDENT_UPDATE', () => get().fetchInitialData());
    socket.on('CRIME_UPDATE', () => get().fetchInitialData());
    
    socket.on('DECISION_COMMAND', (payload) => {
      if (payload.agents_involved?.includes('PoliceAgent')) {
        set((state) => ({ coordinationEvents: [payload, ...state.coordinationEvents].slice(0, 10) }));
        get().fetchInitialData();
      }
    });
  },

  unsubscribeFromSockets: () => {
    const socket = socketService.socket || (typeof socketService.getSocket === 'function' ? socketService.getSocket() : socketService);
    if (!socket || typeof socket.off !== 'function') return;
    socket.off('INCIDENT_UPDATE');
    socket.off('CRIME_UPDATE');
    socket.off('DECISION_COMMAND');
  }
}));