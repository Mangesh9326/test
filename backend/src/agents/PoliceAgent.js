const BaseAgent = require('./BaseAgent');

class PoliceAgent extends BaseAgent {
    constructor() {
        // Expanded resource pool to handle severe security threats and city-wide emergencies
        super('PoliceAgent', 'police', { 
            policeOfficers: 120, 
            patrolVehicles: 45,
            tacticalUnits: 10 
        });
    }

    async evaluate(event) {
        // Track the incident in the agent's isolated memory
        this.memory.activeIncidents.push(event.id);

        // ---------------------------------------------------------
        // SCENARIO 1: Traffic Collision / Accidents
        // ---------------------------------------------------------
        if (event.type === 'ACCIDENT_DETECTED') {
            const officersNeeded = event.severity === 'Critical' ? 6 : 2;
            const vehiclesNeeded = event.severity === 'Critical' ? 3 : 1;

            if (this.allocateResource('policeOfficers', officersNeeded) && this.allocateResource('patrolVehicles', vehiclesNeeded)) {
                this.sendMessage('TrafficAgent', `Police securing perimeter at ${event.location}. Initiating hard closure.`);
                
                return this._createDecision(
                    'Securing Area', event.severity, `Dispatch ${vehiclesNeeded} Patrol Units`, 0.96,
                    { officersDeployed: officersNeeded, patrolVehicles: vehiclesNeeded, intersectionStatus: 'Closed' },
                    `Units deployed to secure accident zone and manage civilian safety at ${event.location}.`,
                    ['Crowd interference', 'Unauthorized civilian entry'], ['TrafficAgent', 'HospitalAgent']
                );
            }
        }

        // ---------------------------------------------------------
        // SCENARIO 2: Crime or Security Breach
        // ---------------------------------------------------------
        if (event.type === 'CRIME_REPORTED' || event.type === 'SECURITY_BREACH') {
            const isCritical = event.severity === 'Critical';
            const officersNeeded = isCritical ? 8 : 2;
            const vehiclesNeeded = isCritical ? 4 : 1;
            const tacticalNeeded = isCritical ? 1 : 0;

            if (
                this.allocateResource('policeOfficers', officersNeeded) && 
                this.allocateResource('patrolVehicles', vehiclesNeeded) &&
                this.allocateResource('tacticalUnits', tacticalNeeded)
            ) {
                if (isCritical) {
                    this.sendMessage('HospitalAgent', `Active security threat at ${event.location}. Put trauma teams on standby.`);
                }

                return this._createDecision(
                    'Active Response', event.severity, isCritical ? 'Deploy Tactical Units' : 'Dispatch Patrol Units', 0.98,
                    { officersDeployed: officersNeeded, patrolVehicles: vehiclesNeeded, tacticalUnits: tacticalNeeded },
                    `Security threat detected at ${event.location}. Units dispatched to neutralize threat and restore order.`,
                    ['Escalation of violence', 'Civilian casualties'], isCritical ? ['HospitalAgent'] : []
                );
            }
        }

        // ---------------------------------------------------------
        // SCENARIO 3: Fire or Gas Leakage (Evacuation Perimeter)
        // ---------------------------------------------------------
        if (event.type === 'FIRE' || event.type === 'GAS_LEAKAGE') {
            const officersNeeded = event.severity === 'Critical' ? 8 : 4;
            const vehiclesNeeded = event.severity === 'Critical' ? 3 : 2;

            if (this.allocateResource('policeOfficers', officersNeeded) && this.allocateResource('patrolVehicles', vehiclesNeeded)) {
                this.sendMessage('FireAgent', `Evacuation perimeter established at ${event.location}. Preventing civilian entry.`);

                return this._createDecision(
                    'Perimeter Control', event.severity, `Establish Evacuation Zone`, 0.95,
                    { officersDeployed: officersNeeded, patrolVehicles: vehiclesNeeded, perimeterStatus: 'Active' },
                    `Hazardous event reported at ${event.location}. Establishing safe perimeter to facilitate fire/hazmat operations.`,
                    ['Civilians trapped in hazard zone'], ['FireAgent', 'TrafficAgent']
                );
            }
        }

        // ---------------------------------------------------------
        // SCENARIO 4: Power Failure (Anti-Looting / Order Maintenance)
        // ---------------------------------------------------------
        if (event.type === 'POWER_FAILURE') {
            const officersNeeded = event.severity === 'Critical' ? 10 : 4;
            const vehiclesNeeded = event.severity === 'Critical' ? 5 : 2;

            if (this.allocateResource('policeOfficers', officersNeeded) && this.allocateResource('patrolVehicles', vehiclesNeeded)) {
                this.sendMessage('UtilityAgent', `Increased police presence in blackout zones at ${event.location} to prevent property crime.`);

                return this._createDecision(
                    'Patrolling', event.severity, `Increase Patrols in Blackout Area`, 0.92,
                    { officersDeployed: officersNeeded, patrolVehicles: vehiclesNeeded, securityStatus: 'Elevated' },
                    `Power failure detected at ${event.location}. Deploying extra patrols to deter looting and maintain public order.`,
                    ['Opportunistic property crime', 'Public panic'], ['UtilityAgent', 'CitizenAgent']
                );
            }
        }

        // ---------------------------------------------------------
        // SCENARIO 5: Severe Weather or Flooding (Evacuation & Shelter)
        // ---------------------------------------------------------
        if (event.type === 'FLOOD' || event.type === 'WEATHER_ALERT') {
            if (event.severity === 'Critical' || event.type === 'FLOOD') {
                const officersNeeded = 6;
                const vehiclesNeeded = 3;

                if (this.allocateResource('policeOfficers', officersNeeded) && this.allocateResource('patrolVehicles', vehiclesNeeded)) {
                    this.sendMessage('CitizenAgent', `Officers dispatched to secure emergency shelters and assist evacuation at ${event.location}.`);

                    return this._createDecision(
                        'Evacuation Support', event.severity, `Assist Citizen Evacuation`, 0.90,
                        { officersDeployed: officersNeeded, patrolVehicles: vehiclesNeeded, shelterSecurity: 'Active' },
                        `Severe weather emergency. Police deployed to assist citizen evacuations and secure temporary shelters.`,
                        ['Looting in evacuated zones', 'Civilians refusing evacuation'], ['CitizenAgent', 'EnvironmentalAgent']
                    );
                }
            }
        }

        // Return null if the event type does not require Police Agent involvement
        return null;
    }
}

module.exports = PoliceAgent;