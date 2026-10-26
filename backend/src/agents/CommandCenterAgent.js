class CommandCenterAgent {
    constructor() {
        this.name = 'CommandCenterAgent';
        this.memory = { cityWideIncidents: [] };
    }

    fuseDecisions(originalEvent, agentDecisions, interAgentMessages) {
        // Track the global incident in memory
        this.memory.cityWideIncidents.push(originalEvent.id);

        // 1. Extract frontend-friendly Actions from each active agent
        const followUpActions = agentDecisions.map(d => {
            const deptName = d.department.charAt(0).toUpperCase() + d.department.slice(1);
            return `${d.action} (${deptName} Agent)`;
        });

        // 2. Resolve Global Priority (Highest severity dictates the global state)
        let finalPriority = 'NOMINAL';
        const severities = agentDecisions.map(d => d.severity);
        
        if (severities.includes('Critical')) {
            finalPriority = 'CRITICAL';
        } else if (severities.includes('High')) {
            finalPriority = 'HIGH';
        } else if (severities.includes('Medium')) {
            finalPriority = 'MEDIUM';
        } else if (severities.includes('Low')) {
            finalPriority = 'LOW';
        }

        // 3. Calculate AI Consensus Score 
        // More participating agents and more inter-agent messages result in a higher confidence score
        const participatingAgents = agentDecisions.length;
        const communicationBonus = interAgentMessages.length * 2;
        let collaborationScore = Math.min((participatingAgents * 12) + communicationBonus + 55, 99); // Capped at 99%
        
        if (participatingAgents === 0) collaborationScore = 100; // No action required = 100% confidence

        // 4. Generate an overarching Executive Directive based on the severity
        let directive = `Coordinated response established for ${originalEvent.type} at ${originalEvent.location}.`;
        
        if (finalPriority === 'CRITICAL') {
            directive = `CRITICAL EMERGENCY: Multi-department task force activated for ${originalEvent.type} at ${originalEvent.location}. Inter-agency protocols engaged.`;
        } else if (finalPriority === 'HIGH') {
            directive = `High Priority Response: Multi-agent coordination active for ${originalEvent.type}.`;
        }

        // Prepend the executive order to the top of the action checklist
        if (participatingAgents > 0) {
            followUpActions.unshift(`Establish Unified Command at ${originalEvent.location} (Command Center)`);
        }

        // 5. Return exact UI Payload structure expected by SocketManager and the Zustand Store
        return {
            eventId: originalEvent.id,
            priority: finalPriority,
            consensus: collaborationScore,
            commandCenterDirective: directive, // Can be used for master dashboard alerts
            actions: followUpActions,          // Populates the Decision Commander checklist
            
            // Extra metadata for logging or future LLM integration
            metadata: {
                participatingDepartments: agentDecisions.map(d => d.department),
                messageVolume: interAgentMessages.length,
                location: originalEvent.location,
                timestamp: originalEvent.timestamp
            }
        };
    }
}

module.exports = CommandCenterAgent;