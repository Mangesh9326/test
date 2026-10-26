const BaseAgent = require("./BaseAgent");

const SIGNAL_LIMITS = Object.freeze({
  MIN_GREEN_TIME: 15,
  MIN_RED_TIME: 15,
  MAX_GREEN_TIME: 90,
  MAX_RED_TIME: 90,
  MAX_GREEN_ADJUSTMENT: 10,
  MAX_RED_ADJUSTMENT: 10,
});
const LEVELS = ["LOW", "MODERATE", "HIGH", "SEVERE", "CRITICAL"];
const LEVEL_SCORES = Object.freeze({
  LOW: 1,
  MODERATE: 2,
  HIGH: 3,
  SEVERE: 4,
  CRITICAL: 5,
});
const INCIDENT_EVENTS = new Set([
  "WEATHER_ALERT",
  "ACCIDENT_DETECTED",
  "FIRE",
  "GAS_LEAKAGE",
  "FLOOD",
  "UTILITY_FAULT",
  "POWER_FAILURE",
  "CRIME_REPORTED",
  "SECURITY_BREACH",
]);
const TRAFFIC_DEMO_EVENTS = new Set(["TRAFFIC_CONGESTION", "TRAFFIC_PREDICTION", "EMERGENCY_ROUTE", "ROAD_CLOSURE"]);

class TrafficAgent extends BaseAgent {
  constructor() {
    super("TrafficAgent", "traffic", {
      trafficOfficers: 50,
      towTrucks: 15,
      roadBarriers: 100,
    });
    this.trafficState = new Map();
  }

  resetAgent() {
    super.resetAgent();
    this.trafficState.clear();
  }

  async evaluate(event) {
    if (event.type === "TRAFFIC_CONDITION")
      return this._evaluateTrafficCondition(event);
    if (event.type === "ROUTE_EVALUATION")
      return this._evaluateRouteEvaluation(event);
    if (TRAFFIC_DEMO_EVENTS.has(event.type))
      return this._evaluateDemoScenario(event);
    if (!INCIDENT_EVENTS.has(event.type)) return null;

    // Incident memory is intentionally separate from ordinary traffic telemetry.
    if (event.id && !this.memory.activeIncidents.includes(event.id))
      this.memory.activeIncidents.push(event.id);

    // SCENARIO 1: Severe Weather (Proactive Traffic Management)
    if (event.type === "WEATHER_ALERT") {
      return this._createDecision(
        "Speed Reduced",
        "Medium",
        "Reduce speed limits by 20km/h",
        0.95,
        {},
        `Slippery roads detected due to ${event.metadata?.weather || "severe weather"}. Proactively slowing traffic to prevent collisions.`,
        ["Increased commute times"],
        [],
      );
    }

    // SCENARIO 2: Traffic Collision / Accidents
    if (event.type === "ACCIDENT_DETECTED") {
      const isMajor =
        event.metadata?.vehicles && event.metadata.vehicles.includes("Bus");
      const officersNeeded = isMajor ? 6 : 2;
      const towTrucksNeeded = isMajor ? 2 : 1;
      if (
        this.allocateResource("trafficOfficers", officersNeeded) &&
        this.allocateResource("towTrucks", towTrucksNeeded)
      ) {
        this.sendMessage(
          "HospitalAgent",
          `Major collision at ${event.location}. Clearing green corridor for EMS.`,
        );
        return this._createDecision(
          "Road Blocked",
          event.severity,
          "Close affected road and initiate diversion",
          0.92,
          {
            officersDeployed: officersNeeded,
            towTrucks: towTrucksNeeded,
            expectedCongestion: isMajor ? "88%" : "42%",
          },
          `Collision involving heavy vehicles. Lanes blocked at ${event.location}. Rerouting active.`,
          ["Secondary collisions", "Gridlock on alternative routes"],
          ["PoliceAgent", "HospitalAgent"],
        );
      }
    }

    // SCENARIO 3: Fire or Gas Leakage (Hazard Zone Diversions)
    if (event.type === "FIRE" || event.type === "GAS_LEAKAGE") {
      const officersNeeded = event.severity === "Critical" ? 6 : 3;
      if (this.allocateResource("trafficOfficers", officersNeeded)) {
        this.sendMessage(
          "PoliceAgent",
          `Traffic diverted around ${event.location}. Handing over inner perimeter control.`,
        );
        return this._createDecision(
          "Roads Closed",
          event.severity,
          "Close nearby intersections and create diversion",
          0.94,
          {
            officersDeployed: officersNeeded,
            diversion: "Active",
            expectedCongestion: "65%",
          },
          `Hazard reported at ${event.location}. Closing surrounding roads to ensure emergency vehicle access and civilian safety.`,
          ["Gridlock on adjacent commercial streets"],
          ["PoliceAgent", "FireAgent", "UtilityAgent"],
        );
      }
    }

    // SCENARIO 4: Flooding (Deploying Physical Barriers)
    if (event.type === "FLOOD") {
      const barriersNeeded = event.severity === "Critical" ? 20 : 10;
      const officersNeeded = event.severity === "Critical" ? 4 : 2;
      if (
        this.allocateResource("roadBarriers", barriersNeeded) &&
        this.allocateResource("trafficOfficers", officersNeeded)
      ) {
        this.sendMessage(
          "EnvironmentalAgent",
          `Roads barricaded at ${event.location} due to high water levels.`,
        );
        return this._createDecision(
          "Route Submerged",
          event.severity,
          "Deploy water barriers and close flooded routes",
          0.96,
          {
            barriersDeployed: barriersNeeded,
            officersDeployed: officersNeeded,
            diversion: "Active",
          },
          `Waterlogging detected at ${event.location}. Deployed physical road barriers to prevent vehicles from entering flooded zones.`,
          ["Stranded vehicles if water rises rapidly"],
          ["EnvironmentalAgent", "CitizenAgent"],
        );
      }
    }

    // SCENARIO 5: Power Failure / Broken Signals
    if (event.type === "UTILITY_FAULT" || event.type === "POWER_FAILURE") {
      const officersNeeded = event.severity === "Critical" ? 4 : 2;
      if (this.allocateResource("trafficOfficers", officersNeeded)) {
        this.sendMessage(
          "UtilityAgent",
          `Officers dispatched to manage intersection manually at ${event.location} while repairs are underway.`,
        );
        return this._createDecision(
          "Manual Control",
          event.severity,
          "Deploy officers for manual intersection control",
          0.9,
          { officersDeployed: officersNeeded, signalStatus: "Offline" },
          `Traffic signals offline at ${event.location} due to power/utility failure. Officers actively directing traffic to prevent gridlock.`,
          ["High risk of minor collisions at intersections"],
          ["UtilityAgent"],
        );
      }
    }

    // SCENARIO 6: Crime or Security Pursuit (Green Corridors)
    if (event.type === "CRIME_REPORTED" || event.type === "SECURITY_BREACH") {
      if (event.severity === "Critical") {
        const officersNeeded = 4;
        const barriersNeeded = 10;
        if (
          this.allocateResource("trafficOfficers", officersNeeded) &&
          this.allocateResource("roadBarriers", barriersNeeded)
        ) {
          this.sendMessage(
            "PoliceAgent",
            "Green corridor activated on Main St. Side intersections blocked to prevent escape.",
          );
          return this._createDecision(
            "Pursuit Support",
            event.severity,
            "Override Signals & Establish Roadblocks",
            0.96,
            {
              officersDeployed: officersNeeded,
              signalOverrides: "Active",
              closedIntersections: 4,
            },
            "Active police pursuit detected. Overriding traffic signal timing to create a green corridor for law enforcement and blocking likely escape routes.",
            ["Civilian traffic gridlock", "Suspect vehicle ramming barricades"],
            ["PoliceAgent"],
          );
        }
      }
    }
    return null;
  }

  _evaluateDemoScenario(event) {
    const metadata = event.metadata || {};
    if (event.type === "EMERGENCY_ROUTE") {
      const intersections = Array.isArray(metadata.intersections) ? metadata.intersections : [];
      return this._createDecision(
        "Simulated Green Corridor",
        event.severity || "CRITICAL",
        "Recommend simulated signal priority along the emergency route",
        0.9,
        { decisionType: "EMERGENCY_ROUTE_SIMULATION", location: event.location, decisionStatus: "SIMULATED", source: "DEMONSTRATION", simulation: true, intersections, payload: { decisionType: "EMERGENCY_ROUTE_SIMULATION", location: event.location, intersections, source: "DEMONSTRATION", simulation: true, outcome: "PENDING" } },
        `SIMULATION ONLY: priority timing is recommended across ${intersections.join(", ") || event.location}; no real Mumbai traffic signal is controlled.`,
        ["Cross-traffic may experience temporary simulated delays."],
        ["HospitalAgent", "PoliceAgent"],
      );
    }
    if (event.type === "ROAD_CLOSURE") {
      const routeRecommendation = this._evaluateRoutes(metadata.routes || [], []);
      return this._createTrafficDecision({ event, decisionType: "ROAD_CLOSURE_DIVERSION", action: routeRecommendation ? `Recommend diversion via ${routeRecommendation.name}` : "Restrict closed road and monitor diversions", analysis: { currentState: "HIGH", reason: "A road closure affects the configured approach." }, prediction: null, densityPrediction: null, signalOptimization: null, routeRecommendation, reasoning: `WHAT: road closure at ${event.location}. WHY: the configured road section is unavailable. DECISION: ${routeRecommendation ? `divert via ${routeRecommendation.name}` : "restrict the affected approach"}. EXPECTED RESULT: avoid the closed road.`, risks: ["Diversion routes may become congested."] });
    }
    return this._evaluateTrafficCondition({ ...event, type: "TRAFFIC_CONDITION" });
  }

  _evaluateTrafficCondition(event) {
    // FIX: Safely parse the location whether it arrives as a string or an object
    let parsedLocation = "";
    if (typeof event.location === 'string') {
        parsedLocation = event.location;
    } else if (event.location && event.location.name) {
        parsedLocation = event.location.name;
    } else if (event.location) {
        parsedLocation = String(event.location);
    }
    
    const location = parsedLocation.trim();

    if (!location || location === "[object Object]") {
      return this._insufficientData("Traffic location is required.", event);
    }
    const analysis = this._analyzeTrafficCondition(event);
    if (!analysis.currentState)
      return this._insufficientData(
        "Traffic level, density, flow, incident impact, road condition, or signal demand is required.",
        event,
      );
    const signal = this._getSignalTiming(event.metadata);
    const stateKey = `${location}|${analysis.currentState}|${signal ? `${signal.currentGreen}|${signal.currentRed}` : "no-signal"}`;
    if (this._isDuplicateDecision(location, stateKey)) return null;

    const prediction = this._predictCongestion(analysis, event.metadata || {});
    const densityPrediction = this._predictDensity(
      analysis,
      event.metadata || {},
    );
    const signalOptimization = this._optimizeSignalTiming(
      analysis.currentState,
      signal,
    );
    const routeRecommendation = this._evaluateRoutes(
      event.metadata?.routes || [],
      event.metadata?.activeIncidents || [],
    );
    const action =
      signalOptimization.recommendation || "Monitor traffic condition";
    const decisionType = signalOptimization.recommendation
      ? "SIGNAL_OPTIMIZATION"
      : "TRAFFIC_ANALYSIS";
    const expectedResult = signalOptimization.recommendation
      ? "Reduce queue buildup on the affected approach while preserving the configured signal cycle."
      : "Maintain the current signal plan while monitoring for a material change.";

    this.trafficState.set(location, {
      stateKey,
      currentState: analysis.currentState,
      updatedAt: Date.now(),
    });
    return this._createTrafficDecision({
      event,
      decisionType,
      action,
      analysis,
      prediction,
      densityPrediction,
      signalOptimization,
      routeRecommendation,
      reasoning: `WHAT: ${analysis.currentState} traffic detected at ${location}. WHY: ${analysis.reason}. DECISION: ${action}. EXPECTED RESULT: ${expectedResult}`,
      risks: signalOptimization.recommendation
        ? [
            "Cross-traffic may experience additional delay.",
            "Traffic may shift to adjacent approaches.",
          ]
        : [],
    });
  }

  _evaluateRouteEvaluation(event) {
    const routeRecommendation = this._evaluateRoutes(
      event.metadata?.routes || [],
      event.metadata?.activeIncidents || [],
    );
    if (!routeRecommendation)
      return this._insufficientData(
        "At least one route with traffic or incident information is required.",
        event,
      );
    return this._createTrafficDecision({
      event,
      decisionType: "ROUTE_RECOMMENDATION",
      action: `Recommend ${routeRecommendation.name}`,
      analysis: {
        currentState: event.metadata?.trafficLevel || "MODERATE",
        reason: routeRecommendation.reason,
      },
      prediction: null,
      densityPrediction: null,
      signalOptimization: null,
      routeRecommendation,
      reasoning: `WHAT: ${routeRecommendation.name} was evaluated against available route conditions. WHY: ${routeRecommendation.reason}. DECISION: Recommend ${routeRecommendation.name}. EXPECTED RESULT: Lower predicted traffic impact for the journey.`,
      risks: [],
    });
  }

  _analyzeTrafficCondition(event) {
    const metadata = event.metadata || {};
    const factors = [
      metadata.trafficLevel,
      metadata.densityLevel,
      metadata.flowLevel,
      metadata.roadCondition,
      metadata.incidentImpact,
      metadata.signalDemand,
    ]
      .map((value) => this._normaliseLevel(value))
      .filter(Boolean);
    const previous = this.trafficState.get(event.location)?.currentState;
    if (previous) factors.push(previous);
    if (!factors.length)
      return {
        currentState: null,
        reason: "No usable traffic factors were supplied.",
      };
    const score = Math.max(...factors.map((level) => LEVEL_SCORES[level]));
    const currentState = LEVELS[score - 1];
    return {
      currentState,
      factors,
      reason: `Assessment combines ${factors.join(", ")} traffic inputs${previous ? " and the previous observed state" : ""}.`,
    };
  }

  _predictCongestion(analysis, metadata) {
    const supplied = this._normaliseLevel(
      metadata.predictedTrafficLevel || metadata.prediction?.predictedState,
    );
    const shouldEscalate =
      ["HIGH", "SEVERE"].includes(analysis.currentState) &&
      ["HIGH", "SEVERE", "CRITICAL"].includes(
        this._normaliseLevel(metadata.signalDemand) || "",
      ) &&
      !supplied;
    const predictedState =
      supplied ||
      (shouldEscalate
        ? LEVELS[
            Math.min(LEVELS.length - 1, LEVEL_SCORES[analysis.currentState])
          ]
        : analysis.currentState);
    return {
      type: "CONGESTION",
      currentState: analysis.currentState,
      predictedState,
      horizonMinutes: Number(metadata.prediction?.horizonMinutes) || 15,
      confidence: 0.8,
      reason: supplied
        ? "Provided by the configured prediction input."
        : `${shouldEscalate ? "Sustained demand may increase congestion" : "Current conditions are expected to persist"} based on deterministic demonstration rules.`,
      source: metadata.prediction?.source || "DEMONSTRATION",
    };
  }

  _predictDensity(analysis, metadata) {
    const currentDensity =
      this._normaliseLevel(metadata.densityLevel) || analysis.currentState;
    const supplied = this._normaliseLevel(metadata.predictedDensityLevel);
    const predictedDensity =
      supplied ||
      (["HIGH", "SEVERE"].includes(currentDensity) &&
      this._normaliseLevel(metadata.signalDemand) === "SEVERE"
        ? "CRITICAL"
        : currentDensity);
    return {
      type: "DENSITY",
      currentState: currentDensity,
      predictedState: predictedDensity,
      horizonMinutes: Number(metadata.densityHorizonMinutes) || 20,
      confidence: 0.75,
      reason: supplied
        ? "Provided by the configured prediction input."
        : "Deterministic demonstration estimate based on density and signal demand.",
      source: metadata.densityPredictionSource || "DEMONSTRATION",
    };
  }

  _getSignalTiming(metadata = {}) {
    const currentGreen = Number(metadata.currentGreenTime);
    const currentRed = Number(metadata.currentRedTime);
    if (
      !Number.isFinite(currentGreen) ||
      !Number.isFinite(currentRed) ||
      currentGreen <= 0 ||
      currentRed <= 0
    )
      return null;
    return {
      currentGreen,
      currentRed,
      signalId: metadata.signalId || null,
      affectedApproach: metadata.affectedApproach || null,
    };
  }

  _optimizeSignalTiming(trafficState, signal) {
    if (!signal)
      return {
        recommendation: null,
        reason: "Signal timing data unavailable",
        status: "INSUFFICIENT_DATA",
      };
    const requestedAdjustment =
      { LOW: 0, MODERATE: 0, HIGH: 5, SEVERE: 8, CRITICAL: 10 }[trafficState] ||
      0;
    const adjustment = Math.max(
      0,
      Math.min(
        requestedAdjustment,
        SIGNAL_LIMITS.MAX_GREEN_ADJUSTMENT,
        SIGNAL_LIMITS.MAX_RED_ADJUSTMENT,
        signal.currentRed - SIGNAL_LIMITS.MIN_RED_TIME,
        SIGNAL_LIMITS.MAX_GREEN_TIME - signal.currentGreen,
      ),
    );
    if (!adjustment)
      return {
        ...signal,
        optimizedGreen: signal.currentGreen,
        optimizedRed: signal.currentRed,
        greenAdjustment: 0,
        redAdjustment: 0,
        recommendation: "Maintain current signal timing",
        reason: "Demand does not justify a safe timing adjustment.",
        status: "RECOMMENDED",
      };
    return {
      ...signal,
      optimizedGreen: signal.currentGreen + adjustment,
      optimizedRed: signal.currentRed - adjustment,
      greenAdjustment: adjustment,
      redAdjustment: -adjustment,
      recommendation: `Recommend GREEN +${adjustment}s / RED ${-adjustment}s`,
      reason: "Higher traffic demand was detected on the affected approach.",
      status: "SIMULATED",
    };
  }

  _evaluateRoutes(routes, activeIncidents) {
    if (!Array.isArray(routes) || !routes.length) return null;
    const assessed = routes
      .map((route, index) => {
        const activeIncident =
          route.incidentImpact === true ||
          route.roadClosed === true ||
          activeIncidents.some(
            (incident) =>
              incident.routeId === route.id || incident.location === route.name,
          );
        const trafficScore =
          LEVEL_SCORES[
            this._normaliseLevel(
              route.predictedTrafficLevel || route.trafficLevel,
            )
          ] || 2;
        return {
          ...route,
          index,
          activeIncident,
          score:
            trafficScore +
            (activeIncident ? 10 : 0) +
            (route.roadClosed ? 10 : 0),
        };
      })
      .sort(
        (left, right) => left.score - right.score || left.index - right.index,
      );
    const best = assessed[0];
    return {
      id: best.id,
      name: best.name || `Route ${String.fromCharCode(65 + best.index)}`,
      reason: best.activeIncident
        ? "Lowest available impact after excluding routes affected by active incidents."
        : "Lower predicted traffic impact from the supplied route conditions.",
      affectedRoutes: assessed
        .filter((route) => route.activeIncident)
        .map(
          (route) =>
            route.name || `Route ${String.fromCharCode(65 + route.index)}`,
        ),
      source: "DEMONSTRATION",
    };
  }

  _createTrafficDecision({
    event,
    decisionType,
    action,
    analysis,
    prediction,
    densityPrediction,
    signalOptimization,
    routeRecommendation,
    reasoning,
    risks,
  }) {
    const payload = {
      decisionType,
      location: event.location,
      currentCondition: analysis.currentState,
      predictedCondition: prediction?.predictedState || null,
      prediction,
      densityPrediction,
      signal: signalOptimization
        ? {
            currentGreen: signalOptimization.currentGreen,
            currentRed: signalOptimization.currentRed,
            optimizedGreen: signalOptimization.optimizedGreen,
            optimizedRed: signalOptimization.optimizedRed,
            greenAdjustment: signalOptimization.greenAdjustment,
            redAdjustment: signalOptimization.redAdjustment,
            signalId: signalOptimization.signalId,
            affectedApproach: signalOptimization.affectedApproach,
          }
        : null,
      routeRecommendation,
      source: event.metadata?.source || "DEMONSTRATION",
      status: signalOptimization?.status || "RECOMMENDED",
      outcome: "PENDING",
    };
    return this._createDecision(
      decisionType === "SIGNAL_OPTIMIZATION"
        ? "Recommended Signal Optimization"
        : "Traffic Intelligence Recommendation",
      event.severity || analysis.currentState,
      action,
      prediction?.confidence || 0.75,
      {
        decisionType,
        location: event.location,
        decisionStatus: payload.status,
        source: payload.source,
        prediction,
        densityPrediction,
        signal: payload.signal,
        routeRecommendation,
        payload,
      },
      reasoning,
      risks,
      [],
    );
  }

  _insufficientData(reason, event) {
    return this._createDecision(
      "Insufficient Traffic Data",
      event.severity || "Low",
      "Request additional traffic telemetry",
      0,
      {
        decisionType: "TRAFFIC_ANALYSIS",
        location: event.location || null,
        decisionStatus: "INSUFFICIENT_DATA",
        source: event.metadata?.source || "DEMONSTRATION",
        outcome: "PENDING",
      },
      reason,
      [],
      [],
    );
  }

  _isDuplicateDecision(location, stateKey) {
    return this.trafficState.get(location)?.stateKey === stateKey;
  }
  _normaliseLevel(value) {
    const normalized = String(value || "")
      .trim()
      .toUpperCase()
      .replace(/VERY[_ -]?HIGH/, "CRITICAL");
    if (LEVEL_SCORES[normalized]) return normalized;
    if (["SLOW", "POOR", "CONGESTED", "INCREASING"].includes(normalized))
      return "HIGH";
    if (["NORMAL", "CLEAR", "GOOD"].includes(normalized)) return "LOW";
    return null;
  }
}

module.exports = TrafficAgent;
