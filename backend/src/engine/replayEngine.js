const db = require('../config/db');
const { broadcastSystemTick } = require('./socketManager');
const agentCoordinator = require('./AgentCoordinator');

let currentSecond = 0;
let simulationInterval = null;
let activeScenarioId = null;
let activeLocationOverride = null;
let activeVehicles = []; // In-memory store for high-performance tick interpolation

/**
 * Calculates smooth interpolation for all active vehicles based on the current simulation tick.
 * This generates the "delta" payload required by the frontend React Leaflet map.
 */

function calculateVehicleDeltas(tickSeconds, activeVehicles) {
    const deltas = {};
    activeVehicles.forEach(v => {
        const route = v.route_path;
        if(route.length < 2) return;
        
        const segment = Math.floor((tickSeconds / 10) % (route.length - 1));
        const factor = (tickSeconds % 10) / 10.0;
        
        deltas[v.vehicle_id] = {
            lat: route[segment][0] + (route[segment+1][0] - route[segment][0]) * factor,
            lng: route[segment][1] + (route[segment+1][1] - route[segment][1]) * factor
        };
    });
    return deltas; // Emitted via Socket.IO: socket.emit('MAP_TICK', deltas)
}

function calculateVehicleMovements(currentTickSeconds, vehicles) {
    const deltaUpdates = {};
    
    vehicles.forEach(vehicle => {
        // Guard clause for vehicles missing valid route paths
        if (!vehicle.route_path || vehicle.route_path.length < 2) return;

        // Determine the current segment of the route (assuming 10s per segment for simulation speed)
        const segmentIndex = Math.floor((currentTickSeconds / 10) % (vehicle.route_path.length - 1));
        const startPoint = vehicle.route_path[segmentIndex];
        const endPoint = vehicle.route_path[segmentIndex + 1];
        
        // Calculate interpolation factor (0.0 to 1.0) for smooth Leaflet marker animation
        const factor = (currentTickSeconds % 10) / 10.0; 
        
        deltaUpdates[vehicle.vehicle_id] = { 
            lat: startPoint[0] + (endPoint[0] - startPoint[0]) * factor, 
            lng: startPoint[1] + (endPoint[1] - startPoint[1]) * factor 
        };
    });
    
    return deltaUpdates; 
}

const startSimulation = async (scenarioId, dynamicLocation) => {
    if (simulationInterval) {
        clearInterval(simulationInterval);
    }
    
    currentSecond = 0;
    activeScenarioId = scenarioId;
    activeLocationOverride = dynamicLocation; 
    
    console.log(`[ENGINE] Starting Scenario ID: ${scenarioId} overridden at Location: ${dynamicLocation}`);

    try {
        // Bulk load vehicle routes into memory ONCE to prevent DB bottlenecking during the tick loop
        const vehiclesResult = await db.query(
            "SELECT vehicle_id, route_path FROM vehicles WHERE status = 'patrolling'"
        );
        activeVehicles = vehiclesResult.rows;
        console.log(`[ENGINE] Loaded ${activeVehicles.length} vehicles for Digital Twin interpolation.`);
    } catch (error) {
        console.error('[ENGINE] Error loading vehicles:', error);
        activeVehicles = [];
    }

    // Initialize the tick-based simulation loop
    simulationInterval = setInterval(async () => {
        currentSecond++;
        await processTick(currentSecond);
    }, 1000);
};

const processTick = async (second) => {
    try {
        // 1. Calculate Vehicle Movements (Digital Twin Delta Update)
        const vehicleUpdates = calculateVehicleMovements(second, activeVehicles);

        // 2. Fetch Active Timeline Events
        const result = await db.query(
            'SELECT id, event_type, payload FROM timeline_events WHERE scenario_id = $1 AND timestamp_second = $2',
            [activeScenarioId, second]
        );
        const events = result.rows;

        // 3. Broadcast System Tick (Now pushing vehicle deltas to Socket.IO)
        // Pass the vehicleUpdates array into the 3rd parameter so the frontend Zustand store can catch it.
        broadcastSystemTick(second, events, vehicleUpdates); 

        // 4. Process Cross-Domain Agent Coordination
        for (const eventRow of events) {
            const rawPayload = eventRow.payload;

            // INJECT THE DYNAMIC LOCATION HERE
            // If the user selected a camera, use that location. Otherwise, fallback to the DB value.
            const finalLocation = activeLocationOverride || rawPayload.location || 'Unknown Location';

            const standardizedEvent = {
                id: `EVT-${activeScenarioId}-${eventRow.id}`,
                timestamp: second,
                type: eventRow.event_type,
                location: finalLocation, // The MAS now receives the exact camera location
                severity: rawPayload.severity || 'Medium',
                metadata: rawPayload 
            };

            await agentCoordinator.processCityEvent(standardizedEvent);
        }

    } catch (error) {
        console.error('[ENGINE] Tick Processing Error:', error);
    }
};

const stopSimulation = () => {
    if (simulationInterval) {
        clearInterval(simulationInterval);
        console.log('[ENGINE] Simulation Stopped.');
    }
};

module.exports = { startSimulation, stopSimulation };