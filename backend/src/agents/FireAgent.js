const BaseAgent = require('./BaseAgent');

class FireAgent extends BaseAgent {
    constructor() {
        // Expanded resource pool to handle Hazmat, rescues, and major fires
        super('FireAgent', 'fire', { 
            fireTrucks: 20, 
            rescueTeams: 15, 
            hazmatUnits: 3,
            waterTankers: 10
        });
    }

    async evaluate(event) {
        // Track the incident in the agent's isolated memory
        this.memory.activeIncidents.push(event.id);

        // ---------------------------------------------------------
        // SCENARIO 1: Fire (e.g., Commercial Building or Residential)
        // ---------------------------------------------------------
        if (event.type === 'FIRE') {
            const trucksNeeded = event.severity === 'Critical' ? 4 : 2;
            const rescueNeeded = event.severity === 'Critical' ? 3 : 1;

            if (this.allocateResource('fireTrucks', trucksNeeded) && this.allocateResource('rescueTeams', rescueNeeded)) {
                
                // Cross-Domain Coordination: Fire dictates actions to Utilities and Police
                this.sendMessage('UtilityAgent', `Critical fire at ${event.location}. Requesting immediate shutdown of local gas and power lines.`);
                this.sendMessage('PoliceAgent', `Fire spreading at ${event.location}. Need 200-meter evacuation perimeter immediately.`);

                return this._createDecision(
                    'Deploying', event.severity, `Dispatch ${trucksNeeded} Fire Engines`, 0.95,
                    { fireTrucksDeployed: trucksNeeded, rescueTeamsDeployed: rescueNeeded, fireRisk: 'Critical' },
                    `Active fire reported. Dispatched fire suppression and rescue teams to ${event.location}.`,
                    ['Structural collapse', 'Explosion risk from nearby utilities'], 
                    ['PoliceAgent', 'UtilityAgent', 'HospitalAgent']
                );
            }
        }

        // ---------------------------------------------------------
        // SCENARIO 2: Gas Leakage or Chemical Spill (Hazmat)
        // ---------------------------------------------------------
        if (event.type === 'GAS_LEAKAGE' || event.type === 'CHEMICAL_SPILL') {
            const hazmatNeeded = event.severity === 'Critical' ? 2 : 1;
            const rescueNeeded = 1;

            if (this.allocateResource('hazmatUnits', hazmatNeeded) && this.allocateResource('rescueTeams', rescueNeeded)) {
                this.sendMessage('PoliceAgent', `Toxic hazard zone at ${event.location}. Establish inner and outer cordons.`);
                
                return this._createDecision(
                    'Hazmat Deployed', event.severity, `Dispatch ${hazmatNeeded} Hazmat Units`, 0.98,
                    { hazmatUnitsDeployed: hazmatNeeded, rescueTeamsDeployed: rescueNeeded, hazardLevel: 'High' },
                    `Chemical/Gas leak detected. Dispatched specialized Hazmat units for containment and decontamination.`,
                    ['Toxic cloud spreading to residential areas', 'Ignition risk'], 
                    ['PoliceAgent', 'HospitalAgent', 'UtilityAgent']
                );
            }
        }

        // ---------------------------------------------------------
        // SCENARIO 3: Traffic Collision / Accidents (Extrication)
        // ---------------------------------------------------------
        if (event.type === 'ACCIDENT_DETECTED') {
            const isMajor = event.metadata?.vehicles && event.metadata.vehicles.includes('Bus');
            
            if (event.severity === 'Critical' || isMajor) {
                // Severe crashes require heavy rescue (Jaws of Life) and fuel spill prevention
                if (this.allocateResource('fireTrucks', 1) && this.allocateResource('rescueTeams', 1)) {
                    return this._createDecision(
                        'Extraction Prep', event.severity, 'Dispatch Rescue & Suppression Unit', 0.92,
                        { fireTrucksDeployed: 1, rescueTeamsDeployed: 1, fuelLeakRisk: 'Elevated' },
                        `Major collision reported. Dispatched unit for vehicle extrication and fuel spill prevention.`,
                        ['Secondary vehicle fires', 'Trapped passengers'], 
                        ['TrafficAgent', 'HospitalAgent']
                    );
                }
            } else {
                // Minor accidents just place the fire agent on standby
                return this._createDecision(
                    'Standby', 'Low', 'Monitor combustion risk', 0.99,
                    { fireRisk: 'Low', fuelLeak: 'Negative', trucksDeployed: 0 },
                    `Assessing minor collision for kinetic fire risks and fuel spillage. Standing by.`,
                    ['Delayed ignition'], 
                    []
                );
            }
        }

        // ---------------------------------------------------------
        // SCENARIO 4: Floods / Severe Weather (Swift-Water Rescue)
        // ---------------------------------------------------------
        if (event.type === 'FLOOD') {
            const rescueNeeded = event.severity === 'Critical' ? 4 : 2;

            if (this.allocateResource('rescueTeams', rescueNeeded)) {
                this.sendMessage('HospitalAgent', `Conducting swift-water rescues at ${event.location}. Prepare for hypothermia and drowning triage.`);
                
                return this._createDecision(
                    'Rescue Operations', event.severity, `Deploy ${rescueNeeded} Water Rescue Teams`, 0.90,
                    { rescueTeamsDeployed: rescueNeeded, waterTankers: 0 },
                    `Flood levels rising. Deployed specialized rescue teams to assist stranded civilians.`,
                    ['Drowning', 'Electrocution from submerged power lines'], 
                    ['HospitalAgent', 'PoliceAgent']
                );
            }
        }

        // Return null for events like Power Failures or Crime, which do not require Fire response
        return null;
    }
}

module.exports = FireAgent;