const socketIo = require('socket.io');

let io;

const initializeSocket = (server) => {
    io = socketIo(server, {
        cors: {
            origin: '*', // In production, restrict this to the frontend URL
            methods: ['GET', 'POST']
        }
    });

    io.on('connection', (socket) => {
        console.log(`[SOCKET] Client connected: ${socket.id}`);
        socket.join('unified_dashboard');

        socket.on('disconnect', () => {
            console.log(`[SOCKET] Client disconnected: ${socket.id}`);
        });
    });

    console.log('[SYSTEM] Socket.IO Initialized for Real-Time Synchronization.');
};

// --- DYNAMIC BROADCAST FUNCTIONS ---

// Broadcasts isolated agent responses to update the 6 department cards dynamically
const broadcastDepartmentUpdate = (updateData) => {
    if (io) io.to('unified_dashboard').emit('DEPARTMENT_UPDATE', updateData);
};

// Broadcasts the fused action plan to the Decision Commander panel
const broadcastDecisionPlan = (plan) => {
    if (io) io.to('unified_dashboard').emit('DECISION_COMMAND', plan);
};

// Broadcasts the live incident to the right sidebar video feed
const broadcastIncidentTrigger = (incidentData) => {
    if (io) io.to('unified_dashboard').emit('INCIDENT_TRIGGERED', incidentData);
};

// Global tick broadcast for the Digital Twin Map interpolation
const broadcastSystemTick = (timestamp, activeEvents, vehicles) => {
    if (io) io.to('unified_dashboard').emit('SYSTEM_TICK', { timestamp, activeEvents, vehicles });
};

module.exports = {
    initializeSocket,
    broadcastDepartmentUpdate,
    broadcastDecisionPlan,
    broadcastIncidentTrigger,
    broadcastSystemTick
};