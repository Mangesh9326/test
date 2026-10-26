const { Pool } = require('pg');
require('dotenv').config();

const pool = new Pool({
    user: process.env.DB_USER,
    host: process.env.DB_HOST,
    database: process.env.DB_NAME,
    password: process.env.DB_PASSWORD,
    port: process.env.DB_PORT,
});

async function seedDatabase() {
    console.log('[SEED] Initializing Master Demonstration Scenario...');

    try {
        // 1. Insert Scenario
        const scenarioResult = await pool.query(`
            INSERT INTO scenarios (name, description, duration_seconds) 
            VALUES ($1, $2, $3) RETURNING id
        `, ['multi_vehicle_heavy_rain.mp4', 'Heavy Rain -> Multi-Vehicle Accident -> Coordinated Response', 120]);
        
        const scenarioId = scenarioResult.rows[0].id;

        // 2. Map your exact timeline to database events
        const events = [
            { second: 0, type: 'SIMULATION_START', payload: { status: 'Nominal', weather: 'Clear' } },
            { second: 5, type: 'WEATHER_ALERT', payload: { weather: 'Heavy Rain', rainfall: '120mm', visibility: 'Low' } },
            { second: 10, type: 'TRAFFIC_UPDATE', payload: { density: 'High', congestion: 'Increasing' } },
            { second: 22, type: 'ACCIDENT_DETECTED', payload: { severity: 'Critical', vehicles: ['Bus', 'Car'], location: 'Ring Road Junction', cctvId: 'CAM-04' } },
            { second: 90, type: 'UTILITY_FAULT', payload: { asset: 'Traffic Signal', status: 'Damaged' } },
            { second: 120, type: 'INCIDENT_RESOLVED', payload: { status: 'Cleared', roadOpen: true } }
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
seedDatabase();