const { Pool } = require('pg');
require('dotenv').config();

const pool = new Pool({
    user: process.env.DB_USER,
    host: process.env.DB_HOST,
    database: process.env.DB_NAME,
    password: process.env.DB_PASSWORD,
    port: process.env.DB_PORT,
});

async function seedRobberyDatabase() {
    console.log('[SEED] Initializing Bank Robbery & City-Wide Pursuit Scenario...');

    try {
        // 1. Insert Scenario
        const scenarioResult = await pool.query(`
            INSERT INTO scenarios (name, description, duration_seconds) 
            VALUES ($1, $2, $3) RETURNING id
        `, ['bank_robbery_pursuit.mp4', 'Bank Robbery -> Suspect Vehicle Escape -> AI Tracking -> Pursuit', 120]);
        
        const scenarioId = scenarioResult.rows[0].id;

        // 2. Map the exact timeline to database events
        const events = [
            { second: 0, type: 'SIMULATION_START', payload: { status: 'Nominal', cityState: 'Normal Operations' } },
            { second: 20, type: 'ANOMALY_DETECTED', payload: { sensor: 'Silent Alarm', status: 'Robbery in Progress', location: 'Central City Bank' } },
            { 
                second: 42, 
                type: 'CRIME_REPORTED', 
                payload: { 
                    severity: 'Critical', 
                    location: 'Central City Bank', 
                    suspects: 2,
                    weapons: true,
                    vehiclesInvolved: ['Black SUV'],
                    cctvId: 'CAM-12',
                    trackingId: 'TRK-8821'
                } 
            },
            { 
                second: 55, 
                type: 'CCTV_TRACKING_UPDATE', 
                payload: { 
                    cctvId: 'CAM-18', 
                    trackingId: 'TRK-8821', 
                    direction: 'Northbound on Main St',
                    confidenceScore: '98%'
                } 
            },
            { 
                second: 65, 
                type: 'CCTV_TRACKING_UPDATE', 
                payload: { 
                    cctvId: 'CAM-23', 
                    trackingId: 'TRK-8821', 
                    direction: 'Approaching Highway On-Ramp',
                    confidenceScore: '96%'
                } 
            },
            { second: 100, type: 'POLICE_INTERCEPT', payload: { status: 'Vehicle Intercepted', location: 'Main St Roadblock' } },
            { second: 120, type: 'INCIDENT_RESOLVED', payload: { status: 'Suspects Arrested', safetyCleared: true } }
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

seedRobberyDatabase();