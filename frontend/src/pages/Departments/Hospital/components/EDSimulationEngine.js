export const SCENARIOS = [
  {
    id: "road",
    name: "Major Road Accident",
    severity: "HIGH",
    patients: 12,
    critical: 3,
    ambulances: 4,
    duration: 360,
    type: "TRAUMA"
  },
  {
    id: "collision",
    name: "Multi-Vehicle Collision",
    severity: "CRITICAL",
    patients: 18,
    critical: 6,
    ambulances: 8,
    duration: 480,
    type: "TRAUMA"
  },
  {
    id: "mass",
    name: "Mass Casualty Incident",
    severity: "CRITICAL",
    patients: 45,
    critical: 15,
    ambulances: 18,
    duration: 720,
    type: "DISASTER"
  },
  {
    id: "cardiac",
    name: "Cardiac Emergency Surge",
    severity: "HIGH",
    patients: 10,
    critical: 5,
    ambulances: 5,
    duration: 360,
    type: "MEDICAL"
  },
  {
    id: "collapse",
    name: "Building Collapse",
    severity: "CRITICAL",
    patients: 28,
    critical: 10,
    ambulances: 12,
    duration: 600,
    type: "DISASTER"
  },
  {
    id: "fire",
    name: "Fire Casualty Surge",
    severity: "HIGH",
    patients: 22,
    critical: 6,
    ambulances: 7,
    duration: 480,
    type: "FIRE"
  },
  {
    id: "industrial",
    name: "Industrial Accident",
    severity: "HIGH",
    patients: 15,
    critical: 5,
    ambulances: 6,
    duration: 420,
    type: "HAZMAT"
  },
  {
    id: "flood",
    name: "Flood Emergency",
    severity: "MODERATE",
    patients: 35,
    critical: 2,
    ambulances: 5,
    duration: 720,
    type: "ENVIRONMENTAL"
  }
];

export const TIMELINE_MILESTONES = [
  {
    timePct: 0.00,
    event: "INCIDENT RECEIVED",
    effect: "Emergency demand baseline established."
  },
  {
    timePct: 0.04,
    event: "EMS DISPATCH",
    effect: "Ambulance resources dispatched to incident."
  },
  {
    timePct: 0.10,
    event: "FIRST PATIENTS ARRIVE",
    effect: "Initial emergency arrivals enter the network."
  },
  {
    timePct: 0.18,
    event: "TRIAGE SURGE",
    effect: "Acuity distribution becomes visible."
  },
  {
    timePct: 0.30,
    event: "QUEUE PRESSURE",
    effect: "Arrival rate begins exceeding available throughput."
  },
  {
    timePct: 0.40,
    event: "HOSPITAL CAPACITY ALERT",
    effect: "Receiving facilities approach operational thresholds."
  },
  {
    timePct: 0.50,
    event: "CAPACITY FORECAST",
    effect: "Near-term emergency department saturation predicted."
  },
  {
    timePct: 0.58,
    event: "DIVERSION RECOMMENDED",
    effect: "Inbound demand redistribution recommended."
  },
  {
    timePct: 0.72,
    event: "THROUGHPUT IMPROVING",
    effect: "Treatment and discharge capacity begin reducing backlog."
  },
  {
    timePct: 0.86,
    event: "QUEUE RECOVERY",
    effect: "Waiting pressure declining across the network."
  },
  {
    timePct: 1.00,
    event: "STABILIZED",
    effect: "Emergency demand returned toward baseline."
  }
];

/* ---------------------------------------------------------
   Utility functions
--------------------------------------------------------- */

const clamp = (value, min, max) =>
  Math.max(min, Math.min(max, value));

const round = value => Math.round(value);

const smoothStep = x => {
  x = clamp(x, 0, 1);
  return x * x * (3 - 2 * x);
};

/*
 * Realistic incident demand curve.

 * Instead of:
 *   0 -> 1 -> 1 -> 0
 *
 * We model:
 *   dispatch delay
 *   rising arrivals
 *   peak arrival period
 *   gradual taper
 *   residual cases
 */
const getIncidentIntensity = (time, duration) => {
  const pct = clamp(time / duration, 0, 1);

  if (pct < 0.05) {
    return 0.15 + (pct / 0.05) * 0.35;
  }

  if (pct < 0.20) {
    return 0.50 + smoothStep((pct - 0.05) / 0.15) * 0.50;
  }

  if (pct < 0.45) {
    return 1.0;
  }

  if (pct < 0.70) {
    return 1.0 - smoothStep((pct - 0.45) / 0.25) * 0.35;
  }

  if (pct < 0.90) {
    return 0.65 - smoothStep((pct - 0.70) / 0.20) * 0.50;
  }

  return 0.15;
};


/*
 * Patient arrivals do not all occur at once.

 * Returns the cumulative proportion of the incident
 * population that has reached hospitals by this point.
 */
const getArrivalProgress = (time, duration) => {
  const pct = clamp(time / duration, 0, 1);

  if (pct < 0.10) {
    return 0.03 + pct * 0.30;
  }

  if (pct < 0.30) {
    return 0.06 + smoothStep((pct - 0.10) / 0.20) * 0.38;
  }

  if (pct < 0.60) {
    return 0.44 + smoothStep((pct - 0.30) / 0.30) * 0.38;
  }

  if (pct < 0.85) {
    return 0.82 + smoothStep((pct - 0.60) / 0.25) * 0.14;
  }

  return 0.96 + smoothStep((pct - 0.85) / 0.15) * 0.04;
};


/*
 * Patient acuity is scenario dependent.
 */
const getAcuityMix = scenario => {
  switch (scenario.type) {
    case "TRAUMA":
      return {
        critical: 0.20,
        high: 0.35,
        moderate: 0.30,
        low: 0.15
      };

    case "DISASTER":
      return {
        critical: 0.28,
        high: 0.34,
        moderate: 0.25,
        low: 0.13
      };

    case "MEDICAL":
      return {
        critical: 0.42,
        high: 0.30,
        moderate: 0.18,
        low: 0.10
      };

    case "FIRE":
      return {
        critical: 0.25,
        high: 0.35,
        moderate: 0.28,
        low: 0.12
      };

    case "HAZMAT":
      return {
        critical: 0.30,
        high: 0.32,
        moderate: 0.25,
        low: 0.13
      };

    case "ENVIRONMENTAL":
      return {
        critical: 0.07,
        high: 0.18,
        moderate: 0.45,
        low: 0.30
      };

    default:
      return {
        critical: 0.15,
        high: 0.30,
        moderate: 0.35,
        low: 0.20
      };
  }
};


/*
 * Deterministic pseudo-random variation.

 * This gives realistic variation without causing the UI
 * to jump differently every render.
 */
const seededVariation = (seed, amount = 1) => {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  const normalized = x - Math.floor(x);

  return 1 + (normalized - 0.5) * amount;
};


/*
 * Estimate how much emergency department throughput
 * a hospital can process during this simulation interval.
 */
const estimateThroughput = hospital => {
  const capacity = Math.max(
    1,
    Number(hospital.emergency_capacity || 20)
  );

  const occupancy = Number(
    hospital.emergency_occupancy || 0
  );

  const occupancyRatio = occupancy / capacity;

  /*
   * ED throughput decreases when the department becomes
   * crowded because staff, beds and diagnostics become bottlenecks.
   */
  let throughputRatio = 0.18;

  if (occupancyRatio >= 0.90) {
    throughputRatio = 0.10;
  } else if (occupancyRatio >= 0.80) {
    throughputRatio = 0.13;
  } else if (occupancyRatio >= 0.70) {
    throughputRatio = 0.16;
  }

  return Math.max(
    1,
    round(capacity * throughputRatio)
  );
};


/*
 * Hospital susceptibility to this incident.

 * A real network does not distribute every incident
 * equally across hospitals.
 */
const getHospitalImpact = (
  hospital,
  index,
  totalHospitals,
  scenario
) => {
  const emergencyCapacity = Number(
    hospital.emergency_capacity || 1
  );

  const normalizedRank =
    index / Math.max(1, totalHospitals - 1);

  /*
   * Larger emergency departments generally absorb
   * more arrivals, but not necessarily all of them.
   */
  let factor = 0.40 - normalizedRank * 0.22;

  /*
   * Existing high occupancy makes a hospital more likely
   * to receive less new demand because EMS diversion kicks in.
   */
  const occupancy =
    Number(hospital.emergency_occupancy || 0) /
    emergencyCapacity;

  if (occupancy > 0.85) {
    factor *= 0.65;
  } else if (occupancy > 0.75) {
    factor *= 0.80;
  }

  /*
   * Trauma incidents preferentially route to facilities
   * with more emergency capacity.
   */
  if (scenario.type === "TRAUMA") {
    factor *= 1.05;
  }

  return clamp(factor, 0.04, 0.45);
};


/* ---------------------------------------------------------
   Main simulation
--------------------------------------------------------- */

export function runSimulation(liveHospitals, scenario, time) {
  const duration = scenario.duration || 360;

  const pct = clamp(time / duration, 0, 1);

  const intensity = getIncidentIntensity(
    time,
    duration
  );

  const previousIntensity = getIncidentIntensity(
    Math.max(0, time - 1),
    duration
  );

  const arrivalProgress = getArrivalProgress(
    time,
    duration
  );

  const previousArrivalProgress =
    getArrivalProgress(
      Math.max(0, time - 1),
      duration
    );


  /* -------------------------------------------------------
     Simulation phase
  ------------------------------------------------------- */

  let phase = "BASELINE";

  if (pct > 0.04 && pct <= 0.20) {
    phase = "ESCALATION";
  } else if (pct > 0.20 && pct <= 0.45) {
    phase = "PEAK";
  } else if (pct > 0.45 && pct <= 0.65) {
    phase = "OVERLOAD";
  } else if (pct > 0.65 && pct <= 0.85) {
    phase = "MITIGATION";
  } else if (pct > 0.85 && pct < 0.97) {
    phase = "RECOVERY";
  } else if (pct >= 0.97) {
    phase = "STABILIZED";
  }


  /* -------------------------------------------------------
     Patient classification
  ------------------------------------------------------- */

  const acuity = getAcuityMix(scenario);

  const cumulativePatients = round(
    scenario.patients * arrivalProgress
  );

  const previousPatients = round(
    scenario.patients * previousArrivalProgress
  );

  const newArrivals = Math.max(
    0,
    cumulativePatients - previousPatients
  );

  const classification = {
    critical: round(
      cumulativePatients * acuity.critical
    ),

    high: round(
      cumulativePatients * acuity.high
    ),

    moderate: round(
      cumulativePatients * acuity.moderate
    ),

    low: round(
      cumulativePatients * acuity.low
    )
  };


  /* -------------------------------------------------------
     Sort hospitals by emergency capacity
  ------------------------------------------------------- */

  const sortedHospitals = [...liveHospitals].sort(
    (a, b) =>
      Number(b.emergency_capacity || 0) -
      Number(a.emergency_capacity || 0)
  );


  /* -------------------------------------------------------
     Hospital simulation
  ------------------------------------------------------- */

  const simulatedHospitals = sortedHospitals.map(
    (hospital, index) => {

      const baseCapacity = Math.max(
        1,
        Number(hospital.emergency_capacity || 1)
      );

      const baseOccupancy = Math.max(
        0,
        Number(hospital.emergency_occupancy || 0)
      );

      const baseQueue = Math.max(
        0,
        Number(hospital.ambulance_queue || 0)
      );

      const baseWait = Math.max(
        0,
        Number(hospital.average_wait_minutes || 0)
      );


      const impact = getHospitalImpact(
        hospital,
        index,
        sortedHospitals.length,
        scenario
      );


      /*
       * How many patients this hospital receives
       * from the incident.
       */
      const hospitalPatients = round(
        scenario.patients *
        arrivalProgress *
        impact *
        seededVariation(index + 10, 0.18)
      );


      const previousHospitalPatients = round(
        scenario.patients *
        previousArrivalProgress *
        impact
      );


      const incrementalPatients =
        Math.max(
          0,
          hospitalPatients -
          previousHospitalPatients
        );


      /*
       * Existing workload affects ability to absorb
       * additional patients.
       */
      const occupancyRatio =
        baseOccupancy / baseCapacity;


      /*
       * Treatment throughput.
       */
      const throughput =
        estimateThroughput(hospital);


      /*
       * Patients discharged/transferred from the ED
       * during the current simulation step.

       * We intentionally keep this conservative so that
       * a hospital can still become overloaded.
       */
      const treatmentCapacity =
        round(
          throughput *
          (0.75 + intensity * 0.25)
        );


      /*
       * Net occupancy change.

       * Arrival adds patients.
       * Treatment/discharge removes patients.
       */
      const grossOccupancyIncrease =
        incrementalPatients;

      const treatmentOutflow =
        pct < 0.10
          ? 0
          : Math.min(
              Math.max(0, baseOccupancy - 2),
              round(
                treatmentCapacity *
                0.18
              )
            );


      const newOccupancy = clamp(
        baseOccupancy +
        grossOccupancyIncrease -
        treatmentOutflow,
        0,
        baseCapacity
      );


      /*
       * Ambulance queue.

       * Critical incidents generate more ambulance
       * arrivals and therefore more pressure.
       */
      const ambulanceArrival =
        round(
          scenario.ambulances *
          impact *
          intensity *
          seededVariation(index + 100, 0.20)
        );


      /*
       * Queue is reduced when treatment capacity becomes
       * available.
       */
      const queueClearance =
        pct > 0.55
          ? round(
              treatmentCapacity * 0.08
            )
          : 0;


      const newQueue = clamp(
        baseQueue +
        ambulanceArrival -
        queueClearance,
        0,
        99
      );


      /*
       * Wait time is based on:

       * - ambulance queue
       * - ED occupancy
       * - existing wait
       * - acuity pressure
       */
      const occupancyPressure =
        Math.max(
          0,
          newOccupancy / baseCapacity - 0.60
        );


      const queuePressure =
        newQueue * 2.5;


      const congestionPenalty =
        occupancyPressure * 45;


      const targetWait =
        baseWait +
        queuePressure +
        congestionPenalty;


      /*
       * Wait time should not instantly jump.
       */
      const newWait = round(
        baseWait +
        (targetWait - baseWait) *
        0.35
      );


      /*
       * Pressure score.

       * This combines more than just occupancy.
       */
      const pressurePercent = clamp(
        round(
          (
            (newOccupancy / baseCapacity) * 70 +
            Math.min(newQueue / 8, 1) * 20 +
            Math.min(newWait / 60, 1) * 10
          )
        ),
        0,
        100
      );


      let level = "NORMAL";

      if (
        pressurePercent >= 90 ||
        newQueue >= 8 ||
        newWait >= 60
      ) {
        level = "CRITICAL";
      } else if (
        pressurePercent >= 75 ||
        newQueue >= 5 ||
        newWait >= 40
      ) {
        level = "HIGH";
      } else if (
        pressurePercent >= 60 ||
        newQueue >= 2 ||
        newWait >= 20
      ) {
        level = "ELEVATED";
      }


      /*
       * Trend considers both arrivals and recovery.
       */
      const netPressure =
        pressurePercent -
        (
          Number(
            hospital.pressure_percent || 0
          )
        );


      let trend = "→ Stable";

      if (netPressure >= 3) {
        trend = "↑ Increasing";
      } else if (netPressure <= -3) {
        trend = "↓ Recovering";
      }


      /*
       * Critical patient estimate for this hospital.
       */
      const hospitalCritical = round(
        hospitalPatients *
        acuity.critical
      );


      return {
        ...hospital,

        simulated: true,

        baseline_occupancy: baseOccupancy,
        baseline_queue: baseQueue,
        baseline_wait: baseWait,

        emergency_occupancy: newOccupancy,

        ambulance_queue: newQueue,

        average_wait_minutes: newWait,

        capacity_level: level,
        emergency_level: level,

        pressure_percent: pressurePercent,

        trend,

        sim_impact: {
          patients: incrementalPatients,
          cumulative_patients: hospitalPatients,
          critical: hospitalCritical,
          arrivals: ambulanceArrival,
          treated: treatmentOutflow,
          wait: Math.max(
            0,
            newWait - baseWait
          )
        }
      };
    }
  );


  /* -------------------------------------------------------
     Network flow
  ------------------------------------------------------- */

  const totalInbound = cumulativePatients;

  const arrivals =
    round(
      newArrivals
    );

  /*
   * Patients progress through the ED over time.
   * These are estimates of patients currently occupying
   * each stage, not mutually exclusive cumulative totals.
   */
  const flow = {

    Arrival: round(
      totalInbound *
      clamp(
        0.12 + intensity * 0.20,
        0.10,
        0.35
      )
    ),

    Registration: round(
      totalInbound *
      clamp(
        0.08 + intensity * 0.06,
        0.06,
        0.16
      )
    ),

    Triage: round(
      totalInbound *
      clamp(
        0.12 + intensity * 0.10,
        0.08,
        0.24
      )
    ),

    Waiting: round(
      totalInbound *
      clamp(
        0.10 + intensity * 0.35,
        0.08,
        0.48
      )
    ),

    Treatment: round(
      totalInbound *
      clamp(
        0.18 + (1 - intensity) * 0.12,
        0.16,
        0.32
      )
    ),

    Observation: round(
      totalInbound *
      clamp(
        0.08 + (1 - intensity) * 0.08,
        0.06,
        0.18
      )
    ),

    Discharge: pct > 0.45
      ? round(
          totalInbound *
          clamp(
            (pct - 0.40) * 0.50,
            0,
            0.28
          )
        )
      : 0
  };


  /* -------------------------------------------------------
     Bottleneck detection
  ------------------------------------------------------- */

  let bottleneck = "ARRIVAL / INBOUND";

  if (
    flow.Waiting >= 5 &&
    flow.Waiting > flow.Triage
  ) {
    bottleneck = "WAITING / QUEUE";
  }

  if (
    flow.Treatment >= 8 &&
    flow.Treatment >= flow.Waiting
  ) {
    bottleneck = "TREATMENT CAPACITY";
  }

  if (
    flow.Triage >= flow.Registration * 1.5 &&
    flow.Triage > 5
  ) {
    bottleneck = "TRIAGE THROUGHPUT";
  }


  /* -------------------------------------------------------
     Network condition
  ------------------------------------------------------- */

  const criticalCount =
    simulatedHospitals.filter(
      h => h.capacity_level === "CRITICAL"
    ).length;

  const highCount =
    simulatedHospitals.filter(
      h => h.capacity_level === "HIGH"
    ).length;


  let networkCondition = "NORMAL";

  if (criticalCount >= 2) {
    networkCondition = "CRITICAL";
  } else if (
    criticalCount === 1 ||
    highCount >= 2
  ) {
    networkCondition = "HIGH";
  } else if (
    highCount === 1 ||
    flow.Waiting >= 8
  ) {
    networkCondition = "ELEVATED";
  }


  /* -------------------------------------------------------
     Alerts
  ------------------------------------------------------- */

  const alerts = [];

  if (criticalCount > 0) {
    alerts.push({
      level: "CRITICAL",
      text:
        `${criticalCount} hospital(s) approaching ` +
        `emergency saturation threshold.`
    });
  }

  if (highCount >= 2) {
    alerts.push({
      level: "HIGH",
      text:
        "Multiple receiving facilities reporting elevated emergency pressure."
    });
  }

  if (flow.Waiting >= 10) {
    alerts.push({
      level: "HIGH",
      text:
        "Emergency waiting queue exceeds operational threshold."
    });
  }

  if (classification.critical >= 5) {
    alerts.push({
      level: "HIGH",
      text:
        "Critical patient volume significantly elevated."
    });
  }

  if (
    pct >= 0.45 &&
    pct < 0.75 &&
    networkCondition !== "NORMAL"
  ) {
    alerts.push({
      level: "MEDIUM",
      text:
        "Network capacity forecast indicates continued pressure."
    });
  }


  /* -------------------------------------------------------
     AI / Hospital Agent
  ------------------------------------------------------- */

  let agent = null;

  if (
    pct >= 0.40 &&
    simulatedHospitals.length > 0
  ) {

    const topHosp =
      [...simulatedHospitals]
        .sort(
          (a, b) =>
            b.pressure_percent -
            a.pressure_percent
        )[0];


    const confidenceBase =
      networkCondition === "CRITICAL"
        ? 94
        : networkCondition === "HIGH"
          ? 89
          : 83;


    const isOverloaded =
      topHosp.pressure_percent >= 85 ||
      topHosp.ambulance_queue >= 6;


    const isRecovering =
      pct >= 0.68;


    let decision =
      "CAPACITY_MONITORING";

    let recommendation =
      "Continue monitoring emergency demand and receiving capacity.";

    let prediction =
      "Demand is expected to remain within the current operational range.";

    let impact =
      "Maintain network stability and monitor receiving facilities.";


    if (isOverloaded && !isRecovering) {

      decision =
        "INBOUND_REDISTRIBUTION";

      recommendation =
        "Redirect selected inbound emergency transports to facilities with available capacity.";

      prediction =
        "Current demand is likely to sustain elevated emergency pressure over the next operational interval.";

      impact =
        "Reduce concentration at overloaded receiving facilities and protect critical-care throughput.";

    } else if (isRecovering) {

      decision =
        "RECOVERY_MONITORING";

      recommendation =
        "Maintain selective diversion while monitoring queue clearance and treatment throughput.";

      prediction =
        "Arrival pressure is declining and emergency capacity is beginning to recover.";

      impact =
        "Support controlled recovery without prematurely returning all inbound demand to the primary facility.";
    }


    agent = {
      condition:
        `${isOverloaded ? "High" : "Elevated"} capacity pressure at ${topHosp.name}`,

      signals:
        `ER occupancy: ${topHosp.pressure_percent}% • ` +
        `Queue: ${topHosp.ambulance_queue} • ` +
        `Wait: ${topHosp.average_wait_minutes}m`,

      prediction,

      decision,

      recommendation,

      impact,

      confidence:
        clamp(
          confidenceBase +
          (isOverloaded ? 2 : 0) -
          (isRecovering ? 3 : 0),
          70,
          97
        ),

      timestamp:
        `T+${String(time).padStart(3, "0")}s`,

      source:
        "Hospital Agent • Simulation"
    };
  }


  /* -------------------------------------------------------
     Timeline
  ------------------------------------------------------- */

  const timeline =
    TIMELINE_MILESTONES
      .filter(
        milestone =>
          pct >= milestone.timePct
      )
      .reverse();


  return {
    hospitals: simulatedHospitals,

    phase,

    flow,

    bottleneck,

    classification,

    networkCondition,

    alerts,

    agent,

    timeline,

    progress: pct * 100,

    /*
     * Additional simulation telemetry.
     * Useful for charts/debugging but does not break
     * existing consumers.
     */
    simulation: {
      intensity,
      previousIntensity,
      arrivalProgress,
      newArrivals: arrivals,
      cumulativePatients: totalInbound,
      remainingPatients:
        Math.max(
          0,
          scenario.patients - totalInbound
        )
    }
  };
}