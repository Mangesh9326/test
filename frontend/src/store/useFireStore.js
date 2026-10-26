// frontend/src/store/useFireStore.js
import { create } from 'zustand';

export const useFireStore = create((set, get) => ({
  incidents: [],
  summary: { activeIncidents: 0, critical: 0, highPriority: 0, onScene: 0, contained: 0, avgResponseTime: 0 },
  selectedIncident: null,
  isLoading: true,
  isSimulation: true,
  filters: {
    search: '',
    severity: 'ALL',
    status: 'ALL',
    incident_type: 'ALL',
    ward: 'ALL'
  },

  setFilters: (newFilters) => {
    set((state) => ({ filters: { ...state.filters, ...newFilters } }));
    get().fetchIncidents();
  },

  setSelectedIncident: (incident) => set({ selectedIncident: incident }),

  fetchSummary: async () => {
    try {
      const res = await fetch('/api/fire/incidents/summary');
      const data = await res.json();
      set({ summary: data });
    } catch (error) {
      console.error("[Fire Store] Failed to fetch summary", error);
    }
  },

  fetchIncidents: async () => {
    set({ isLoading: true });
    try {
      const { search, severity, status, incident_type, ward } = get().filters;
      const params = new URLSearchParams();
      if (search) params.append('search', search);
      if (severity !== 'ALL') params.append('severity', severity);
      if (status !== 'ALL') params.append('status', status);
      if (incident_type !== 'ALL') params.append('incident_type', incident_type);
      if (ward !== 'ALL') params.append('ward', ward);

      const res = await fetch(`/api/fire/incidents?${params.toString()}`);
      const data = await res.json();
      set({ incidents: data, isLoading: false });
    } catch (error) {
      console.error("[Fire Store] Failed to fetch incidents", error);
      set({ isLoading: false });
    }
  },

  initialize: () => {
    get().fetchSummary();
    get().fetchIncidents();
  }
}));