const { Pool } = require('pg');
require('dotenv').config({ path: '../.env' }); // Ensure this points to your backend .env

const pool = new Pool({
    user: process.env.DB_USER || 'smart_city_user',
    host: process.env.DB_HOST || 'localhost',
    database: 'smart_city_db',
    password: 'Mangesh9326',
    port: process.env.DB_PORT || 5432,
});

// Real Mumbai Hospitals (25)
const REAL_HOSPITALS = [
    { name: 'Lilavati Hospital, Bandra', lat: 19.0510, lng: 72.8277, beds: 314, icu: 45 },
    { name: 'KEM Hospital, Parel', lat: 19.0028, lng: 72.8415, beds: 1800, icu: 120 },
    { name: 'Nanavati Hospital, Vile Parle', lat: 19.0963, lng: 72.8398, beds: 350, icu: 50 },
    { name: 'Hinduja Hospital, Mahim', lat: 19.0336, lng: 72.8384, beds: 400, icu: 60 },
    { name: 'Bombay Hospital, Marine Lines', lat: 18.9405, lng: 72.8285, beds: 725, icu: 100 },
    { name: 'Breach Candy Hospital', lat: 18.9670, lng: 72.8050, beds: 212, icu: 35 },
    { name: 'Jaslok Hospital, Pedder Road', lat: 18.9717, lng: 72.8090, beds: 350, icu: 45 },
    { name: 'Sion Hospital (LTMG)', lat: 19.0360, lng: 72.8617, beds: 1900, icu: 150 },
    { name: 'Hiranandani Hospital, Powai', lat: 19.1227, lng: 72.9157, beds: 240, icu: 40 },
    { name: 'Kokilaben Hospital, Andheri', lat: 19.1314, lng: 72.8249, beds: 750, icu: 110 },
    { name: 'Tata Memorial Hospital', lat: 19.0048, lng: 72.8427, beds: 600, icu: 80 },
    { name: 'Fortis Hospital, Mulund', lat: 19.1643, lng: 72.9431, beds: 300, icu: 45 },
    { name: 'Holy Family Hospital, Bandra', lat: 19.0543, lng: 72.8315, beds: 268, icu: 38 },
    { name: 'Bhatia Hospital, Tardeo', lat: 18.9645, lng: 72.8131, beds: 210, icu: 30 },
    { name: 'Saifee Hospital, Charni Road', lat: 18.9525, lng: 72.8183, beds: 250, icu: 37 },
    { name: 'Cooper Hospital, Juhu', lat: 19.1065, lng: 72.8364, beds: 636, icu: 75 },
    { name: 'Global Hospital, Parel', lat: 18.9950, lng: 72.8402, beds: 450, icu: 55 },
    { name: 'Wockhardt Hospital, Mumbai Central', lat: 18.9754, lng: 72.8236, beds: 350, icu: 40 },
    { name: 'SL Raheja Hospital, Mahim', lat: 19.0435, lng: 72.8437, beds: 170, icu: 25 },
    { name: 'Zenith Hospital, Malad', lat: 19.1868, lng: 72.8465, beds: 120, icu: 15 },
    { name: 'Rajawadi Hospital, Ghatkopar', lat: 19.0792, lng: 72.8986, beds: 500, icu: 50 },
    { name: 'Bhagwati Hospital, Borivali', lat: 19.2452, lng: 72.8524, beds: 300, icu: 30 },
    { name: 'Shatabdi Hospital, Kandivali', lat: 19.2016, lng: 72.8461, beds: 320, icu: 40 },
    { name: 'GT Hospital, Fort', lat: 18.9455, lng: 72.8329, beds: 520, icu: 60 },
    { name: 'St. George Hospital, Fort', lat: 18.9392, lng: 72.8385, beds: 460, icu: 50 }
];

// Real Mumbai Police Stations (20)
const REAL_POLICE = [
    { name: 'Colaba Police Station', lat: 18.9150, lng: 72.8250 },
    { name: 'Marine Drive Police Station', lat: 18.9351, lng: 72.8235 },
    { name: 'Bandra Police Station', lat: 19.0560, lng: 72.8320 },
    { name: 'BKC Police Station', lat: 19.0620, lng: 72.8640 },
    { name: 'Andheri Police Station', lat: 19.1170, lng: 72.8465 },
    { name: 'Dharavi Police Station', lat: 19.0445, lng: 72.8550 },
    { name: 'Dadar Police Station', lat: 19.0195, lng: 72.8425 },
    { name: 'Powai Police Station', lat: 19.1230, lng: 72.9055 },
    { name: 'Juhu Police Station', lat: 19.1025, lng: 72.8285 },
    { name: 'Khar Police Station', lat: 19.0710, lng: 72.8350 },
    { name: 'Worli Police Station', lat: 19.0005, lng: 72.8160 },
    { name: 'Ghatkopar Police Station', lat: 19.0860, lng: 72.9085 },
    { name: 'Kurla Police Station', lat: 19.0665, lng: 72.8825 },
    { name: 'Borivali Police Station', lat: 19.2295, lng: 72.8565 },
    { name: 'Chembur Police Station', lat: 19.0525, lng: 72.9005 },
    { name: 'Malad Police Station', lat: 19.1855, lng: 72.8485 },
    { name: 'Goregaon Police Station', lat: 19.1630, lng: 72.8490 },
    { name: 'Mulund Police Station', lat: 19.1725, lng: 72.9560 },
    { name: 'Sakinaka Police Station', lat: 19.1005, lng: 72.8885 },
    { name: 'Tardeo Police Station', lat: 18.9735, lng: 72.8140 }
];

// Land boundaries to prevent entities spawning in the sea
const LAND_ZONES = [
    { minLat: 18.915, maxLat: 18.970, minLng: 72.815, maxLng: 72.838 }, // South
    { minLat: 18.980, maxLat: 19.030, minLng: 72.825, maxLng: 72.855 }, // Central
    { minLat: 19.040, maxLat: 19.230, minLng: 72.825, maxLng: 72.860 }, // West Suburbs
    { minLat: 19.060, maxLat: 19.220, minLng: 72.870, maxLng: 72.930 }  // East Suburbs
];

function getRandomLandCoord() {
    const zone = LAND_ZONES[Math.floor(Math.random() * LAND_ZONES.length)];
    return {
        lat: (Math.random() * (zone.maxLat - zone.minLat) + zone.minLat).toFixed(6),
        lng: (Math.random() * (zone.maxLng - zone.minLng) + zone.minLng).toFixed(6)
    };
}

function generateRoute(steps) {
    const start = getRandomLandCoord();
    let route = [[parseFloat(start.lat), parseFloat(start.lng)]];
    for (let i = 0; i < steps; i++) {
        route.push([route[i][0] + (Math.random() - 0.5) * 0.008, route[i][1] + (Math.random() - 0.5) * 0.005]);
    }
    return JSON.stringify(route);
}

async function seedDigitalTwin() {
    const client = await pool.connect();
    try {
        await client.query('BEGIN');
        await client.query('TRUNCATE map_entities, vehicles, incidents CASCADE');
        
        let queries = [];

        // 1. Inject Real Hospitals
        REAL_HOSPITALS.forEach(h => {
            queries.push(`('hospital', '${h.name}', ${h.lat}, ${h.lng}, '{"capacity": ${h.beds}, "icuBeds": ${h.icu}, "emergencyStatus": "Normal"}', 'health')`);
        });

        // 2. Inject Real Police Stations
        REAL_POLICE.forEach(p => {
            queries.push(`('police_station', '${p.name}', ${p.lat}, ${p.lng}, '{"status": "Active", "patrolsAvailable": 5}', 'police')`);
        });

        // 3. Generate Requested Quantities procedurally on land
        const gen = (type, prefix, count, meta, layer) => {
            for(let i=0; i<count; i++) {
                const c = getRandomLandCoord();
                queries.push(`('${type}', '${prefix} ${i}', ${c.lat}, ${c.lng}, '${JSON.stringify(meta)}', '${layer}')`);
            }
        };
        
        gen('traffic_signal', 'Signal', 120, { phase: "Green" }, 'traffic');
        gen('cctv', 'Cam', 80, { status: "Live", aiDetection: "Active" }, 'traffic');
        gen('fire_station', 'Fire Dept', 15, { engines: 4 }, 'fire');
        gen('street_light', 'Light Pin', 250, { status: "ON" }, 'utilities');
        gen('transformer', 'Grid Node', 80, { load: "74%" }, 'utilities');
        gen('water_sensor', 'Water Valve', 100, { pressure: "Normal" }, 'utilities');
        gen('weather_station', 'Mausam Node', 40, { temp: "28C", aqi: 110 }, 'weather');

        await client.query(`INSERT INTO map_entities (entity_type, name, latitude, longitude, metadata, layer_group) VALUES ${queries.join(', ')}`);

        // 4. Generate Vehicles
        let vehicles = [];
        const genVehicles = (type, dept, count) => {
            for(let i=0; i<count; i++) {
                const c = getRandomLandCoord();
                vehicles.push(`('${type}', '${dept}', ${c.lat}, ${c.lng}, '${generateRoute(12)}', 'patrolling')`);
            }
        };
        genVehicles('ambulance', 'health', 60);
        genVehicles('police_vehicle', 'police', 80);
        genVehicles('fire_truck', 'fire', 40);
        genVehicles('public_bus', 'transit', 100);

        await client.query(`INSERT INTO vehicles (vehicle_type, department, current_lat, current_lng, route_path, status) VALUES ${vehicles.join(', ')}`);

        await client.query('COMMIT');
        console.log("Mumbai Digital Twin Successfully Seeded with Real Anchors.");
    } finally {
        client.release();
    }
}
seedDigitalTwin();