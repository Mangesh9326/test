import { create } from 'zustand';
import socketService from '../services/socketService';

export const useAlertsStore = create((set, get) => ({
  alerts: [],
  readAlertIds: new Set(),
  isLoading: true,
  error: null,
  userLocation: { latitude: 19.0178, longitude: 72.8478 }, 
  
  fetchAlerts: async () => {
    set({ isLoading: true, error: null });
    const { userLocation } = get();
    try {
      const res = await fetch(`/api/citizen/alerts?latitude=${userLocation.latitude}&longitude=${userLocation.longitude}`);
      if (!res.ok) throw new Error("Unable to load alerts.");
      const data = await res.json();
      set({ alerts: data.alerts || [], isLoading: false });
    } catch (error) {
      set({ error: error.message, isLoading: false });
    }
  },

  markAsRead: (alertId) => {
    set((state) => {
      const updated = new Set(state.readAlertIds);
      updated.add(alertId);
      return { readAlertIds: updated };
    });
  },

  askAI: async (message, alertContext) => {
    try {
      const res = await fetch('/api/citizen/assistant', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message, 
          context: { alert: alertContext, location: get().userLocation } 
        })
      });
      if (!res.ok) throw new Error("AI unavailable");
      const data = await res.json();
      return data.reply;
    } catch (err) {
      return "The available data does not provide a confirmed reason or the AI assistant is temporarily unavailable.";
    }
  },

 subscribeToSockets: () => {
  // Safely extract the raw socket instance from the wrapper service
  const socket =
    socketService.socket ||
    (typeof socketService.getSocket === 'function'
      ? socketService.getSocket()
      : socketService);

  if (!socket || typeof socket.on !== 'function') {
    console.warn(
      "[Alerts Store] Socket instance missing .on(). Check socketService.js implementation."
    );
    return;
  }

  socket.on('DEPARTMENT_UPDATE', () => get().fetchAlerts());
  socket.on('INCIDENT_TRIGGERED', () => get().fetchAlerts());
},

unsubscribeFromSockets: () => {
  const socket =
    socketService.socket ||
    (typeof socketService.getSocket === 'function'
      ? socketService.getSocket()
      : socketService);

  if (!socket || typeof socket.off !== 'function') return;

  socket.off('DEPARTMENT_UPDATE');
  socket.off('INCIDENT_TRIGGERED');
}
}));