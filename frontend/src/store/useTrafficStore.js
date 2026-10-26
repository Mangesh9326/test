import { create } from 'zustand';

export const useTrafficStore = create((set) => ({
  activeVehicles: 0,
  congestionLevel: 'Low',
  incidents: [],
  
  updateTrafficData: (data) => set((state) => ({
    ...state,
    ...data
  })),
}));