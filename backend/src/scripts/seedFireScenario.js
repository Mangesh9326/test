const { Pool } = require('pg');
require('dotenv').config();

const pool = new Pool({
    user: process.env.DB_USER,
    host: process.env.DB_HOST,
    database: process.env.DB_NAME,
    password: process.env.DB_PASSWORD,
    port: process.env.DB_PORT,
});

async function seedFireDatabase() {
    console.log('[SEED] Initializing Building Fire Demonstration Scenario...');

    try {
        // 1. Insert Scenario
        const scenarioResult = await pool.query(`
            INSERT INTO scenarios (name, description, duration_seconds) 
            VALUES ($1, $2, $3) RETURNING id
        `, ['commercial_building_fire.mp4', 'Commercial Building Fire -> Multi-Department Emergency Response', 140]);
        
        const scenarioId = scenarioResult.rows[0].id;

        // 2. Map the exact timeline to database events
        const events = [
            { second: 0, type: 'SIMULATION_START', payload: { status: 'Nominal', grid: 'Stable' } },
            { second: 20, type: 'ANOMALY_DETECTED', payload: { sensor: 'CCTV & Smoke Detector', status: 'Smoke Detected', location: 'Central Commercial Complex' } },
            { 
                second: 30, 
                type: 'FIRE', 
                payload: { 
                    severity: 'Critical', 
                    location: 'Central Commercial Complex', 
                    buildingType: 'Commercial',
                    windSpeed: '15 km/h',
                    cctvId: 'CAM-12' 
                } 
            },
            { second: 130, type: 'FIRE_CONTAINED', payload: { status: 'Intensity Decreasing', smokeLevel: 'Low' } },
            { second: 140, type: 'INCIDENT_RESOLVED', payload: { status: 'Extinguished', safetyCleared: true } }
        ];

        for (const event of events) {
            await pool.query(`
                INSERT INTO timeline_events (scenario_id, timestamp_second, event_type, payload) 
                VALUES ($1, $2, $3, $4)
            `, [scenarioId, event.second, event.type, JSON.stringify(event.payload)]);
        }

        console.log(`[SEED] Success! Inserted ${events.length} events for Scenario ID: ${scenarioId}`);
    } catch (error) {
        console.error('[SEED ERROR]', error);
    } finally {
        await pool.end();
    }
}

seedFireDatabase();