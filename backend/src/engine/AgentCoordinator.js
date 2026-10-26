const eventBus = require('./EventBus');
const { broadcastDecisionPlan, broadcastDepartmentUpdate, broadcastIncidentTrigger } = require('./socketManager');

// Import Domain Agents
const TrafficAgent = require('../agents/TrafficAgent');
const HospitalAgent = require('../agents/HospitalAgent');
const PoliceAgent = require('../agents/PoliceAgent');
const FireAgent = require('../agents/FireAgent');
const UtilityAgent = require('../agents/UtilityAgent');
const EnvironmentalAgent = require('../agents/EnvironmentalAgent');
const CitizenAgent = require('../agents/CitizenAgent');
const CommandCenterAgent = require('../agents/CommandCenterAgent');

class AgentCoordinator {
    constructor() {
        // Initialize all independent agents with their isolated resource pools
        this.agents = [
            new TrafficAgent(),
            new HospitalAgent(),
            new PoliceAgent(),
            new FireAgent(),
            new UtilityAgent(),
            new EnvironmentalAgent(),
            new CitizenAgent()
        ];
        
        this.commandCenter = new CommandCenterAgent();
        this.messageLog = [];

        // Track inter-agent communication for the Command Center payload
        eventBus.on('INTER_AGENT_MESSAGE', (msg) => {
            this.messageLog.push(msg);
            console.log(`[EVENT BUS] ${msg.from} -> ${msg.to}: ${msg.message}`);
        });
    }

    getAgent(name) {
        return this.agents.find((agent) => agent.name === name) || null;
    }

    async processCityEvent(eventPayload) {
        console.log(`[COORDINATOR] Ingesting Dynamic Event: ${eventPayload.type} at ${eventPayload.location} (Severity: ${eventPayload.severity})`);
        
        // 1. Trigger the Live Incident Feed on the frontend immediately
        if (eventPayload.severity === 'Critical' || eventPayload.severity === 'High') {
            broadcastIncidentTrigger({
                id: eventPayload.id,
                type: eventPayload.type,
                message: `${eventPayload.type} Detected`,
                location: eventPayload.location
            });
        }

        // Reset the message log for this specific evaluation cycle
        this.messageLog = [];

        // 2. Parallel Evaluation: All agents evaluate the dynamic event independently simultaneously
        const agentPromises = this.agents.map(agent => agent.evaluate(eventPayload));
        const results = await Promise.all(agentPromises);

        // 3. Filter out agents that determined the event was outside their domain (returned null)
        const activeDecisions = results.filter(decision => decision !== null);

        if (activeDecisions.length === 0) {
            console.log(`[COORDINATOR] Event ${eventPayload.id} required no agent action.`);
            return null;
        }

        // 4. Dynamic Dashboard Push: Broadcast each active agent's isolated state to its respective UI card
        activeDecisions.forEach(decision => {
            if (decision.department) {
                broadcastDepartmentUpdate({
                    department: decision.department,    // e.g., 'traffic', 'hospital'
                    status: decision.status,            // e.g., 'Blocked', 'Deploying'
                    severity: decision.severity,        // e.g., 'Critical', 'High'
                    metadata: decision.metadata         // Dynamic KPIs (e.g., congestion %, available beds)
                });
            }
        });

        // 5. Decision Fusion: Command Center merges the independent decisions and inter-agent messages
        // We delay the final fusion slightly (e.g., 2 seconds) to simulate AI processing time and 
        // allow the UI to stagger the department updates before the final plan drops.
        setTimeout(() => {
            const finalCoordinatedPlan = this.commandCenter.fuseDecisions(
                eventPayload, 
                activeDecisions, 
                this.messageLog
            );

            // 6. Broadcast Output: Push the structured JSON to the React Decision Commander via Socket.IO
            broadcastDecisionPlan(finalCoordinatedPlan);
            console.log(`[COORDINATOR] Unified Plan Broadcasted. Priority: ${finalCoordinatedPlan.priority}`);
            
        }, 2000); 

        return true;
    }
}

module.exports = new AgentCoordinator();
