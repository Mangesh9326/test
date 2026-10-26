const BaseAgent = require('./BaseAgent');

class UtilityAgent extends BaseAgent {
    constructor() {
        // Expanded resource pool to handle floods, fires, and blackouts
        super('UtilityAgent', 'utility', { 
            repairCrews: 30, 
            engineers: 15, 
            waterPumps: 20, 
            mobileGenerators: 10 
        });
    }

    async evaluate(event) {
        // Track the incident in the agent's isolated memory
        this.memory.activeIncidents.push(event.id);

        // ---------------------------------------------------------
        // SCENARIO 1: Isolated Infrastructure Fault (e.g., Broken Signal)
        // ---------------------------------------------------------
        if (event.type === 'UTILITY_FAULT') {
            const crewsNeeded = 1;
            
            if (this.allocateResource('repairCrews', crewsNeeded)) {
                this.sendMessage('TrafficAgent', `Traffic signal offline at ${event.location}. Deploy manual officers.`);
                
                return this._createDecision(
                    'Fault Detected', event.severity, `Dispatch Repair Crew`, 0.91,
                    { repairCrews: crewsNeeded, asset: event.metadata?.asset || 'Infrastructure', gridStatus: 'Nominal' },
                    `Infrastructure damage detected. Crew dispatched for repairs.`,
                    ['Extended outage'], ['TrafficAgent']
                );
            }
        }

        // ---------------------------------------------------------
        // SCENARIO 2: Commercial Building Fire or Gas Leak
        // ---------------------------------------------------------
        if (event.type === 'FIRE' || event.type === 'GAS_LEAKAGE') {
            const engineersNeeded = 2; // Needed to safely shut down grids
            
            if (this.allocateResource('engineers', engineersNeeded)) {
                this.sendMessage('FireAgent', `Gas and power isolated in a 500m radius around ${event.location}. Safe for fire suppression.`);
                
                return this._createDecision(
                    'Grid Isolated', 'Critical', `Shut down local power/gas grid`, 0.95,
                    { engineersDeployed: engineersNeeded, gridStatus: 'Isolated' },
                    `Shutting down local gas and power grid to prevent secondary explosions at ${event.location}.`,
                    ['Extended power outage for local residents'], ['FireAgent']
                );
            }
        }

        // ---------------------------------------------------------
        // SCENARIO 3: Power Failure / City Blackout
        // ---------------------------------------------------------
        if (event.type === 'POWER_FAILURE') {
            const crewsNeeded = 2;
            const engineersNeeded = 1;
            
            if (this.allocateResource('repairCrews', crewsNeeded) && this.allocateResource('engineers', engineersNeeded)) {
                this.sendMessage('TrafficAgent', `Grid failure at ${event.location}. Traffic signals may be offline.`);
                this.sendMessage('HospitalAgent', `Grid failure near hospital zone. Confirm backup generator status.`);
                
                return this._createDecision(
                    'Outage Response', event.severity, `Dispatch Repair Crews & Engineers`, 0.92,
                    { repairCrews: crewsNeeded, engineersDeployed: engineersNeeded, gridStatus: 'Offline' },
                    `Major grid failure detected. Crews dispatched to identify and repair the fault at ${event.location}.`,
                    ['City-wide cascade failure'], ['TrafficAgent', 'HospitalAgent']
                );
            }
        }

        // ---------------------------------------------------------
        // SCENARIO 4: Severe Weather or Flooding
        // ---------------------------------------------------------
        if (event.type === 'FLOOD' || event.type === 'WEATHER_ALERT') {
            // Only deploy physical assets if the weather is severe or flooding is confirmed
            if (event.type === 'FLOOD' || event.severity === 'Critical' || event.severity === 'High') {
                const crewsNeeded = 3;
                const pumpsNeeded = 5;
                
                if (this.allocateResource('repairCrews', crewsNeeded) && this.allocateResource('waterPumps', pumpsNeeded)) {
                    return this._createDecision(
                        'Flood Mitigation', event.severity, `Deploy Water Pumps to Substations`, 0.88,
                        { repairCrews: crewsNeeded, waterPumps: pumpsNeeded, gridStatus: 'At Risk' },
                        `Severe weather/flood risk. Deployed pumps to protect critical underground power substations at ${event.location}.`,
                        ['Substation flooding', 'Electrocution hazard'], ['EnvironmentalAgent']
                    );
                }
            } else {
                // Low severity weather just puts the grid on standby mode
                return this._createDecision(
                    'Standby', 'Low', `Monitor Infrastructure`, 0.98,
                    { gridStatus: 'Stable' },
                    `Monitoring weather impact on power lines and drainage systems.`,
                    ['Minor localized outages'], []
                );
            }
        }

        // ---------------------------------------------------------
        // SCENARIO 5: Traffic Collision / Accidents
        // ---------------------------------------------------------
        if (event.type === 'ACCIDENT_DETECTED') {
            // Check if it's a severe accident that might have hit a utility pole or transformer
            if (event.severity === 'Critical') {
                return this._createDecision(
                    'Standby', 'Medium', `Monitor Grid Integrity`, 0.90,
                    { gridStatus: 'Nominal' },
                    `Critical accident reported at ${event.location}. Standing by in case of structural damage to utility poles.`,
                    ['Concealed infrastructure damage'], ['PoliceAgent', 'TrafficAgent']
                );
            }
        }

        // Return null if the event type does not require Utility Agent involvement
        return null;
    }
}

module.exports = UtilityAgent;