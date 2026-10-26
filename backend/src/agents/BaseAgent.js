const eventBus = require('../engine/EventBus');

class BaseAgent {
    constructor(name, department, initialResources) {
        this.name = name;
        this.department = department; // Mapped to the frontend UI cards
        
        // Deep copy the initial resources so we can restore them later during a reset
        this.initialResources = JSON.parse(JSON.stringify(initialResources));
        this.resources = JSON.parse(JSON.stringify(initialResources));
        
        this.memory = {
            activeIncidents: [],
            pastIncidents: [],
            workload: 0,
            recentDecisions: []
        };
        this.messagesReceived = [];

        // Listen for direct inter-agent messages via the Event Bus
        eventBus.on('INTER_AGENT_MESSAGE', (payload) => {
            if (payload.to === this.name || payload.to === 'ALL') {
                this.messagesReceived.push(payload);
            }
        });
    }

    // ---------------------------------------------------------
    // RESOURCE & LIFECYCLE MANAGEMENT
    // ---------------------------------------------------------

    allocateResource(resourceType, amount) {
        if (this.resources[resourceType] !== undefined && this.resources[resourceType] >= amount) {
            this.resources[resourceType] -= amount;
            this.memory.workload += amount;
            return true;
        }
        console.warn(`[${this.name}] Resource allocation failed for ${amount}x ${resourceType}. Available: ${this.resources[resourceType]}`);
        return false;
    }

    // Return resources to the pool when an incident is resolved
    releaseResource(resourceType, amount) {
        if (this.resources[resourceType] !== undefined) {
            this.resources[resourceType] += amount;
            // Ensure workload doesn't drop below zero
            this.memory.workload = Math.max(0, this.memory.workload - amount);
            return true;
        }
        return false;
    }

    // Move an incident from active to past memory
    resolveIncident(eventId) {
        const index = this.memory.activeIncidents.indexOf(eventId);
        if (index > -1) {
            this.memory.activeIncidents.splice(index, 1);
            this.memory.pastIncidents.push(eventId);
        }
    }

    // Completely wipe the agent's state (Used by ReplayEngine between scenarios)
    resetAgent() {
        this.resources = JSON.parse(JSON.stringify(this.initialResources));
        this.memory = {
            activeIncidents: [],
            pastIncidents: [],
            workload: 0,
            recentDecisions: []
        };
        this.messagesReceived = [];
        console.log(`[${this.name}] Agent state and resources reset for new simulation.`);
    }

    // ---------------------------------------------------------
    // COMMUNICATION & DECISION OUTPUT
    // ---------------------------------------------------------

    sendMessage(targetAgent, message) {
        eventBus.emit('INTER_AGENT_MESSAGE', {
            from: this.name,
            to: targetAgent,
            message: message,
            timestamp: Date.now()
        });
    }

    // Abstract method to be overridden by child domain agents
    async evaluate(event) {
        throw new Error(`[${this.name}] evaluate() method not implemented.`);
    }

    // Formats the decision exactly for the React Dashboard dynamic UI
    _createDecision(status, severity, action, confidence, resources, reasoning, risks, collaboratingAgents) {
        this.memory.recentDecisions.push({ timestamp: Date.now(), reasoning });
        
        return {
            agent: this.name,
            department: this.department, 
            status: status,
            severity: severity,
            action: action,
            metadata: {
                ...resources,
                confidence: `${(confidence * 100).toFixed(0)}%`,
                reasoning: reasoning,
                risks: risks,
                collaborators: collaboratingAgents
            }
        };
    }
}

module.exports = BaseAgent;