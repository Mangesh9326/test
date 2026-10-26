// frontend/src/store/useFireDispatchStore.js
import { create } from 'zustand';

export const useFireDispatchStore = create((set, get) => ({
  queue: [],
  summary: {},
  stations: [],
  isLoading: true,
  selectedIncident: null,
  filters: {
    search: '',
    severity: 'ALL',
    status: 'ALL',
    active_only: true
  },

  setFilters: (newFilters) => {
    set((state) => ({ filters: { ...state.filters, ...newFilters } }));
    get().fetchData();
  },

  setSelectedIncident: (incident) => set({ selectedIncident: incident }),

  fetchData: async () => {
    set({ isLoading: true });
    try {
      const { search, severity, status, active_only } = get().filters;
      
      const params = new URLSearchParams();
      if (search) params.append('search', search);
      if (severity !== 'ALL') params.append('severity', severity);
      if (status !== 'ALL') params.append('status', status);
      if (active_only) params.append('active_only', 'true');

      const [queueRes, summaryRes, stationsRes] = await Promise.all([
        fetch(`/api/fire/dispatch?${params.toString()}`),
        fetch('/api/fire/dispatch/summary'),
        fetch('/api/fire/dispatch/stations')
      ]);

      const [queueData, summaryData, stationsData] = await Promise.all([
        queueRes.json(),
        summaryRes.json(),
        stationsRes.json()
      ]);

      set({ 
        queue: queueData, 
        summary: summaryData, 
        stations: stationsData, 
        isLoading: false 
      });
    } catch (error) {
      console.error("[Fire Dispatch Store] Data fetch error", error);
      set({ isLoading: false });
    }
  }
}));