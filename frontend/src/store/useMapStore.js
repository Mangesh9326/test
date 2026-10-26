import { create } from 'zustand';

export const useMapStore = create((set, get) => ({
    entities: [],
    vehicles: [],
    incidents: [],
    
    // UI Controls
    activeLayers: { health: true, police: true, traffic: true, utilities: true, fire: true, weather: true },
    searchQuery: '',
    statusFilter: 'ALL', // 'ALL', 'NORMAL', 'CRITICAL'

    initializeMap: (data) => set({ entities: data.static, vehicles: data.vehicles, incidents: data.incidents }),
    
    // Delta updates for performance (called via Socket.IO)
    updateVehicles: (deltas) => set(state => ({
        vehicles: state.vehicles.map(v => 
            deltas[v.vehicle_id] ? { ...v, current_lat: deltas[v.vehicle_id].lat, current_lng: deltas[v.vehicle_id].lng } : v
        )
    })),

    setSearch: (query) => set({ searchQuery: query }),
    toggleLayer: (layer) => set(state => ({
        activeLayers: { ...state.activeLayers, [layer]: !state.activeLayers[layer] }
    })),

    // Filter Logic Selector
    getFilteredEntities: () => {
        const { entities, activeLayers, searchQuery } = get();
        return entities.filter(e => {
            const matchesLayer = activeLayers[e.layer_group];
            const matchesSearch = e.name.toLowerCase().includes(searchQuery.toLowerCase()) || 
                                  e.entity_type.toLowerCase().includes(searchQuery.toLowerCase());
            return matchesLayer && matchesSearch;
        });
    }
}));