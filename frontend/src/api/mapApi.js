// src/api/mapApi.js
export const fetchMapEntities = async () => {
    const response = await fetch('/api/map/entities');
    const data = await response.json();
    // Returns { staticEntities: [...], vehicles: [...], incidents: [...] }
    return data; 
};