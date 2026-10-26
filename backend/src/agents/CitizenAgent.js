const BaseAgent = require('./BaseAgent');

class CitizenAgent extends BaseAgent {
    constructor() {
        // Expanded resource pool for mass communications and call centers
        super('CitizenAgent', 'citizen', { 
            broadcastBandwidth: 100, // Percentage of emergency network capacity
            callCenterAgents: 50     // Operators available for 311/911 civilian inquiries
        });
    }

    async evaluate(event) {
        // Track the incident in the agent's isolated memory
        this.memory.activeIncidents.push(event.id);

        // ---------------------------------------------------------
        // SCENARIO 1: Severe Weather or Flooding (Mass Alert)
        // ---------------------------------------------------------
        if (event.type === 'WEATHER_ALERT' || event.type === 'FLOOD') {
            const bandwidthNeeded = event.severity === 'Critical' ? 40 : 15;
            const agentsNeeded = event.severity === 'Critical' ? 20 : 5;

            if (this.allocateResource('broadcastBandwidth', bandwidthNeeded) && this.allocateResource('callCenterAgents', agentsNeeded)) {
                return this._createDecision(
                    'Transmitting', event.severity, 'Broadcast Mass Weather/Flood Alert', 0.98,
                    { broadcastBandwidthUsed: `${bandwidthNeeded}%`, callCenterAgentsActive: agentsNeeded },
                    `Dispatched mass mobile alerts regarding environmental hazards at ${event.location}. Call center staffed for civilian inquiries.`,
                    ['Network congestion', 'Public panic'], ['EnvironmentalAgent', 'PoliceAgent']
                );
            }
        }

        // ---------------------------------------------------------
        // SCENARIO 2: Fire, Gas Leak, or Chemical Spill (Evacuation)
        // ---------------------------------------------------------
        if (event.type === 'FIRE' || event.type === 'GAS_LEAKAGE' || event.type === 'CHEMICAL_SPILL') {
            const bandwidthNeeded = 20;

            if (this.allocateResource('broadcastBandwidth', bandwidthNeeded)) {
                return this._createDecision(
                    'Evacuation Alert', event.severity, 'Issue Geo-fenced Evacuation Order', 0.99,
                    { broadcastBandwidthUsed: `${bandwidthNeeded}%`, reach: 'Nearby Residents' },
                    `Life-threatening hazard at ${event.location}. Issued localized push notifications instructing immediate evacuation.`,
                    ['Evacuation stampedes'], ['PoliceAgent', 'FireAgent']
                );
            }
        }

        // ---------------------------------------------------------
        // SCENARIO 3: Crime or Security Breach (Lockdown)
        // ---------------------------------------------------------
        if (event.type === 'CRIME_REPORTED' || event.type === 'SECURITY_BREACH') {
            const bandwidthNeeded = 10;

            if (this.allocateResource('broadcastBandwidth', bandwidthNeeded)) {
                return this._createDecision(
                    'Security Alert', event.severity, 'Issue Shelter-in-Place Warning', 0.97,
                    { broadcastBandwidthUsed: `${bandwidthNeeded}%`, reach: '1,500 Civilians' },
                    `Active security threat at ${event.location}. Broadcasted shelter-in-place orders to localized mobile devices.`,
                    ['Civilian interference with police operations'], ['PoliceAgent']
                );
            }
        }

        // ---------------------------------------------------------
        // SCENARIO 4: Power Failure (Information Update)
        // ---------------------------------------------------------
        if (event.type === 'POWER_FAILURE') {
            const agentsNeeded = 20; // High call volume expected during blackouts

            if (this.allocateResource('callCenterAgents', agentsNeeded) && this.allocateResource('broadcastBandwidth', 5)) {
                return this._createDecision(
                    'Information Update', event.severity, 'Send Outage Status Notifications', 0.95,
                    { callCenterAgentsActive: agentsNeeded, broadcastBandwidthUsed: '5%' },
                    `Power outage at ${event.location}. Sent SMS updates with estimated restoration times to prevent call center overload.`,
                    ['Call center capacity breached'], ['UtilityAgent']
                );
            }
        }

        // ---------------------------------------------------------
        // SCENARIO 5: Traffic Collision / Accidents (Route Diversion)
        // ---------------------------------------------------------
        if (event.type === 'ACCIDENT_DETECTED') {
            if (event.severity === 'Critical' || event.severity === 'High') {
                if (this.allocateResource('broadcastBandwidth', 5)) {
                    return this._createDecision(
                        'Traffic Alert', event.severity, 'Issue Route Diversion Suggestion', 0.92,
                        { broadcastBandwidthUsed: '5%', reach: 'Commuters in Transit' },
                        `Major collision at ${event.location}. Pushed alternate route suggestions to civilian navigation apps to ease gridlock.`,
                        ['Alternative routes becoming congested'], ['TrafficAgent']
                    );
                }
            }
        }

        // Return null if the event type does not require Citizen Agent involvement
        return null;
    }
}

module.exports = CitizenAgent;