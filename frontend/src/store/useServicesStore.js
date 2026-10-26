import { create } from 'zustand';
import socketService from '../services/socketService';

export const useServicesStore = create((set, get) => ({
  services: [],
  isLoading: true,
  error: null,
  isSimulated: true,
  userLocation: { latitude: 19.0178, longitude: 72.8478, name: "Dadar Junction (Demo)" }, 
  searchQuery: '',
  selectedCategory: 'All',

  setSearchQuery: (query) => {
    set({ searchQuery: query });
    get().fetchServices();
  },

  setSelectedCategory: (category) => {
    set({ selectedCategory: category });
    get().fetchServices();
  },
  
  fetchServices: async () => {
    set({ isLoading: true, error: null });
    const { userLocation, searchQuery, selectedCategory } = get();
    
    try {
      const params = new URLSearchParams({
        latitude: userLocation.latitude,
        longitude: userLocation.longitude,
        radius: 10
      });
      
      if (searchQuery) params.append('search', searchQuery);
      if (selectedCategory !== 'All') params.append('category', selectedCategory);

      const res = await fetch(`/api/citizen/services?${params.toString()}`);
      if (!res.ok) throw new Error("Unable to load city services.");
      
      const data = await res.json();
      set({ 
        services: data.services || [], 
        isSimulated: data.isSimulated,
        isLoading: false 
      });
    } catch (error) {
      set({ error: error.message, isLoading: false, services: [] });
    }
  },

  askAI: async (message, serviceContext) => {
    try {
      const res = await fetch('/api/citizen/assistant', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message, 
          context: { service: serviceContext, location: get().userLocation } 
        })
      });
      if (!res.ok) throw new Error("AI unavailable");
      const data = await res.json();
      return data.reply;
    } catch (err) {
      return "The AI assistant is temporarily unavailable. Please try again later.";
    }
  },

  subscribeToSockets: () => {
    const socket = socketService.socket || (typeof socketService.getSocket === 'function' ? socketService.getSocket() : socketService);
    if (!socket || typeof socket.on !== 'function') {
      console.warn("[Services Store] Socket instance missing .on(). Check socketService.js implementation.");
      return;
    }
    socket.on('DEPARTMENT_UPDATE', () => get().fetchServices());
    socket.on('DECISION_COMMAND', () => get().fetchServices());
  },
  
  unsubscribeFromSockets: () => {
    const socket = socketService.socket || (typeof socketService.getSocket === 'function' ? socketService.getSocket() : socketService);
    if (!socket || typeof socket.off !== 'function') return;
    socket.off('DEPARTMENT_UPDATE');
    socket.off('DECISION_COMMAND');
  }
}));