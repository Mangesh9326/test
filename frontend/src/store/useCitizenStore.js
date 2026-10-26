import { create } from 'zustand';
import socketService from '../services/socketService';

export const useCitizenStore = create((set, get) => ({
  location: { latitude: 19.0178, longitude: 72.8478, name: "Dadar Junction (Demo)" }, 
  radius: 5,
  services: [],
  incidents: [],
  traffic: [],
  alerts: [],
  isSimulated: true,
  isLoading: true,
  error: null,
  
  setLocation: (lat, lng, name = "Selected Location") => {
    set({ location: { latitude: lat, longitude: lng, name } });
    get().fetchAroundMe();
  },

  setRadius: (radius) => {
    set({ radius });
    get().fetchAroundMe();
  },

  fetchAroundMe: async () => {
    set({ isLoading: true, error: null });
    const { location, radius } = get();
    try {
      const res = await fetch(`/api/citizen/around-me?latitude=${location.latitude}&longitude=${location.longitude}&radius=${radius}`);
      if (!res.ok) throw new Error("Unable to load nearby city information.");
      const data = await res.json();
      
      set({ 
        services: data.services,
        incidents: data.incidents,
        traffic: data.traffic,
        alerts: data.alerts,
        isSimulated: data.isSimulated,
        isLoading: false 
      });
    } catch (error) {
      console.error("[Citizen Store] Initialization error:", error);
      set({ isLoading: false, error: error.message });
    }
  },

  askAI: async (message) => {
    const state = get();
    const context = {
      location: state.location,
      radius: state.radius,
      nearbyIncidents: state.incidents,
      nearbyServices: state.services,
      alerts: state.alerts
    };

    try {
      const res = await fetch('/api/citizen/assistant', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, context })
      });
      if (!res.ok) throw new Error("AI unavailable");
      const data = await res.json();
      return data.reply;
    } catch (err) {
      return "The AI assistant is temporarily unavailable. Please try again later.";
    }
  },

  subscribeToSockets: () => {
    // Safely extract the raw socket instance from the wrapper service
    const socket = socketService.socket || (typeof socketService.getSocket === 'function' ? socketService.getSocket() : socketService);

    if (!socket || typeof socket.on !== 'function') {
      console.warn("[Citizen Store] Socket instance missing .on(). Check socketService.js implementation.");
      return;
    }

    socket.on('DEPARTMENT_UPDATE', () => get().fetchAroundMe());
    socket.on('DECISION_COMMAND', () => get().fetchAroundMe());
  },

  unsubscribeFromSockets: () => {
    const socket = socketService.socket || (typeof socketService.getSocket === 'function' ? socketService.getSocket() : socketService);

    if (!socket || typeof socket.off !== 'function') return;

    socket.off('DEPARTMENT_UPDATE');
    socket.off('DECISION_COMMAND');
  }
}));