const BaseAgent = require("./BaseAgent");

class EnvironmentalAgent extends BaseAgent {
  constructor() {
    // Expanded resource pool for comprehensive environmental monitoring
    super("EnvironmentalAgent", "citizen", { 
      monitoringDrones: 12,
      environmentalInspectors: 8,
      airQualitySensors: 20
    });
  }

  async evaluate(event) {
    // Track the incident in the agent's isolated memory
    this.memory.activeIncidents.push(event.id);

    // ---------------------------------------------------------
    // SCENARIO 1: Severe Weather (Heavy Rain / Storms)
    // ---------------------------------------------------------
    if (event.type === "WEATHER_ALERT") {
      this.sendMessage("TrafficAgent", `Heavy rain detected. Risk of hydroplaning high.`);
      this.sendMessage("CitizenAgent", `Initiate severe weather broadcasts.`);

      return this._createDecision(
        "Alerting",
        event.severity,
        "Issue Severe Weather Advisory",
        0.99,
        {
          environment: event.metadata?.weather || "Adverse Weather",
          rainfall: event.metadata?.rainfall || "Unknown",
        },
        `Severe weather detected. Alerting integrated departments to adjust baseline thresholds.`,
        ["Flash flooding", "Reduced visibility"],
        ["TrafficAgent", "CitizenAgent"]
      );
    }

    // ---------------------------------------------------------
    // SCENARIO 2: Flooding
    // ---------------------------------------------------------
    if (event.type === "FLOOD") {
      const dronesNeeded = event.severity === "Critical" ? 4 : 2;
      const inspectorsNeeded = 2;

      if (
        this.allocateResource("monitoringDrones", dronesNeeded) &&
        this.allocateResource("environmentalInspectors", inspectorsNeeded)
      ) {
        this.sendMessage("UtilityAgent", `Flood waters rising at ${event.location}. Protect underground substations.`);
        this.sendMessage("TrafficAgent", `Waterlogging detected at ${event.location}. Divert low-clearance vehicles.`);

        return this._createDecision(
          "Flood Assessment",
          event.severity,
          "Deploy aerial drones for water level mapping",
          0.94,
          {
            monitoringDrones: dronesNeeded,
            environmentalInspectors: inspectorsNeeded,
            floodRisk: "Elevated"
          },
          `Deployed aerial drones to map flood plain expansion and dispatched inspectors to assess infrastructure risk.`,
          ["Substation inundation", "Rapid water level rise"],
          ["UtilityAgent", "TrafficAgent"]
        );
      }
    }

    // ---------------------------------------------------------
    // SCENARIO 3: Fire (Smoke Plume Monitoring)
    // ---------------------------------------------------------
    if (event.type === "FIRE") {
      const dronesNeeded = event.severity === "Critical" ? 3 : 2;

      if (this.allocateResource("monitoringDrones", dronesNeeded)) {
        this.sendMessage("CitizenAgent", `Smoke plume spreading East from ${event.location}. Update evacuation radius.`);

        return this._createDecision(
          "Monitoring Spread",
          event.severity,
          "Assess wind direction and smoke spread",
          0.92,
          {
            monitoringDrones: dronesNeeded,
            windSpeed: event.metadata?.windSpeed || "15 km/h",
            aqiImpact: "Moderate"
          },
          `Deployed drones to estimate toxic smoke spread based on current wind vectors.`,
          ["Wind shifts pushing smoke into residential zones", "Poor air quality"],
          ["CitizenAgent", "FireAgent"]
        );
      }
    }

    // ---------------------------------------------------------
    // SCENARIO 4: Gas Leakage / Chemical Spill (AQI Monitoring)
    // ---------------------------------------------------------
    if (event.type === "GAS_LEAKAGE" || event.type === "CHEMICAL_SPILL") {
      const sensorsNeeded = 5;
      const dronesNeeded = 2;

      if (
        this.allocateResource("airQualitySensors", sensorsNeeded) &&
        this.allocateResource("monitoringDrones", dronesNeeded)
      ) {
        this.sendMessage("PoliceAgent", `Toxic plume tracking active at ${event.location}. Align perimeter with wind direction.`);
        this.sendMessage("HospitalAgent", `High probability of inhalation injuries from ${event.location}.`);

        return this._createDecision(
          "Hazmat Tracking",
          event.severity,
          "Deploy portable AQI sensors and mapping drones",
          0.96,
          {
            airQualitySensors: sensorsNeeded,
            monitoringDrones: dronesNeeded,
            toxicityLevel: "Hazardous"
          },
          `Hazardous leak detected. Deploying real-time air quality sensors and aerial mapping to track the toxic cloud.`,
          ["Unpredictable toxic spread", "Mass civilian exposure"],
          ["PoliceAgent", "HospitalAgent"]
        );
      }
    }

    // Return null if the event type does not require Environmental Agent involvement
    return null;
  }
}

module.exports = EnvironmentalAgent;