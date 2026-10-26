const express = require("express");
const router = express.Router();
const pool = require("../../config/db");
const agentCoordinator = require("../../engine/AgentCoordinator");

const severityOrder = { CRITICAL: 0, SEVERE: 1, HIGH: 2, MODERATE: 3, LOW: 4 };
const colorBySeverity = {
  CRITICAL: "text-red-500",
  SEVERE: "text-red-500",
  HIGH: "text-orange-400",
  MODERATE: "text-yellow-400",
  LOW: "text-emerald-400",
};

// In-memory, cross-device scenario playback state. This is intentionally ephemeral
// (not persisted to the database) — it just lets one device (e.g. the Simulate
// Demonstration control page) drive what another device (e.g. a Live Traffic
// display) is currently showing, via simple polling. Restarting the server
// resets it. If this needs to survive restarts or run across multiple server
// instances, move it into a table/Redis key instead of module state.
let playbackState = {
  scenarioId: null,
  isRunning: false,
  isPaused: false,
  updatedAt: null,
};

async function enrichSignalTiming(event) {
  const metadata = event.metadata || {};
  if (
    Number.isFinite(Number(metadata.currentGreenTime)) &&
    Number.isFinite(Number(metadata.currentRedTime))
  )
    return event;
  const identifier = metadata.signalId || event.location;
  if (!identifier) return event;
  const result = await pool.query(
    `SELECT entity_id, name, metadata FROM map_entities WHERE entity_type = 'traffic_signal' AND (entity_id::text = $1 OR name = $1) LIMIT 1`,
    [identifier],
  );
  const signal = result.rows[0];
  if (!signal) return event;
  return {
    ...event,
    metadata: {
      ...metadata,
      signalId: metadata.signalId || signal.entity_id,
      currentGreenTime: metadata.currentGreenTime ?? signal.metadata.green_time,
      currentRedTime: metadata.currentRedTime ?? signal.metadata.red_time,
      trafficLevel: metadata.trafficLevel || signal.metadata.traffic_level,
    },
  };
}

async function persistDecision(decision) {
  if (!decision) return null;
  const metadata = decision.metadata || {};
  const payload = metadata.payload || metadata;
  const result = await pool.query(
    `
    INSERT INTO traffic_decisions (agent, decision_type, location, severity, recommendation, reason, payload, status, source, outcome)
    VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10)
    RETURNING *`,
    [
      decision.agent,
      metadata.decisionType || "TRAFFIC_ANALYSIS",
      metadata.location || null,
      decision.severity,
      decision.action,
      metadata.reasoning || null,
      JSON.stringify(payload),
      metadata.decisionStatus || "RECOMMENDED",
      metadata.source || "DEMONSTRATION",
      metadata.outcome || "PENDING",
    ],
  );
  return result.rows[0];
}

function decisionResponse(decision, record) {
  if (!decision)
    return {
      status: "NO_CHANGE",
      reason:
        "An equivalent traffic condition is already being monitored; no duplicate recommendation was generated.",
    };
  return {
    status: decision.metadata?.decisionStatus || "RECOMMENDED",
    decision,
    record,
  };
}

function scenarioResponse(row) {
  return {
    id: row.id,
    name: row.name,
    type: row.type,
    location: row.location,
    coordinates: { lat: Number(row.latitude), lng: Number(row.longitude) },
    severity: row.severity,
    description: row.description,
    trafficLevel: row.traffic_level,
    densityLevel: row.density_level,
    flowLevel: row.flow_level,
    affectedApproach: row.affected_approach,
    source: row.source,
    isDemo: row.is_demo,
    event: row.event_data,
  };
}

async function getDemoScenario(id) {
  const result = await pool.query(
    "SELECT * FROM traffic_demo_scenarios WHERE id = $1 AND is_demo = true AND source = 'DEMONSTRATION'",
    [id],
  );
  return result.rows[0] || null;
}

// GET /api/traffic/scenarios - database is the source of truth for traffic demonstrations.
router.get("/scenarios", async (req, res) => {
  try {
    const result = await pool.query(
      "SELECT * FROM traffic_demo_scenarios WHERE is_demo = true AND source = 'DEMONSTRATION' ORDER BY id ASC",
    );
    res.json({ scenarios: result.rows.map(scenarioResponse) });
  } catch (error) {
    console.error("Traffic scenarios API error:", error);
    res
      .status(500)
      .json({ error: "Failed to fetch traffic demonstration scenarios." });
  }
});

// GET /api/traffic/scenarios/:id - full event payload needed for deterministic replay.
router.get("/scenarios/:id", async (req, res) => {
  try {
    const scenario = await getDemoScenario(req.params.id);
    if (!scenario)
      return res
        .status(404)
        .json({ error: "Traffic demonstration scenario not found." });
    res.json({ scenario: scenarioResponse(scenario) });
  } catch (error) {
    console.error("Traffic scenario API error:", error);
    res
      .status(500)
      .json({ error: "Failed to fetch traffic demonstration scenario." });
  }
});

// POST /api/traffic/scenarios/:id/replay - evaluates exactly the event stored in the database.
router.post("/scenarios/:id/replay", async (req, res) => {
  try {
    const scenario = await getDemoScenario(req.params.id);
    if (!scenario)
      return res
        .status(404)
        .json({ error: "Traffic demonstration scenario not found." });
    const eventData = scenario.event_data || {};
    const event = {
      id: `traffic-demo:${scenario.id}:${Date.now()}`,
      type: eventData.type || scenario.type,
      location: eventData.location || scenario.location,
      severity: eventData.severity || scenario.severity,
      metadata: {
        ...(eventData.metadata || {}),
        source: "DEMONSTRATION",
        isDemo: true,
      },
    };
    const decision = await agentCoordinator
      .getAgent("TrafficAgent")
      .evaluate(event);
    const record = await persistDecision(decision);
    // A missing decision here means the agent evaluated the event and found
    // NO_CHANGE (an equivalent condition is already being monitored) — that's
    // a valid outcome, not a failure, so this responds 200 rather than 422.
    // decisionResponse() already sets status: "NO_CHANGE" in that case.
    res.status(decision ? 201 : 200).json({
      scenario: scenarioResponse(scenario),
      ...decisionResponse(decision, record),
    });
  } catch (error) {
    console.error("Traffic scenario replay error:", error);
    res
      .status(500)
      .json({ error: "Unable to replay the traffic demonstration scenario." });
  }
});

// GET /api/traffic/playback-state - current cross-device scenario playback state.
// Polled by any Live Traffic display so it can auto-mirror whatever the Simulate
// Demonstration control page (on this device or another) last set.
router.get("/playback-state", (req, res) => {
  res.json(playbackState);
});

// POST /api/traffic/playback-state - set the active scenario/playback state.
// Called by the Simulate Demonstration control page on Start/Pause/Reset.
// Body: { scenarioId: string|null, isRunning?: boolean, isPaused?: boolean }
// Sending scenarioId as null/"" clears it (equivalent to Reset to Manual Exploration Mode).
router.post("/playback-state", (req, res) => {
  const { scenarioId, isRunning, isPaused } = req.body || {};

  if (scenarioId === null || scenarioId === undefined || scenarioId === "") {
    playbackState = {
      scenarioId: null,
      isRunning: false,
      isPaused: false,
      updatedAt: new Date().toISOString(),
    };
    return res.json(playbackState);
  }

  playbackState = {
    scenarioId: String(scenarioId),
    isRunning: !!isRunning,
    isPaused: !!isPaused,
    updatedAt: new Date().toISOString(),
  };
  res.json(playbackState);
});

// GET /api/traffic/cameras
router.get("/cameras", async (req, res) => {
  try {
    const result = await pool.query(
      `SELECT entity_id as camera_id, name as camera_code, latitude, longitude, metadata->>'status' as status, metadata->>'road_name' as road_name, metadata->>'direction' as direction, metadata->>'stream_url' as stream_url FROM map_entities WHERE entity_type = 'cctv'`,
    );
    res.json(result.rows);
  } catch (error) {
    console.error("Traffic Cameras API Error:", error);
    res.status(500).json({ error: "Failed to fetch traffic cameras" });
  }
});

// GET /api/traffic/live - seeded signal telemetry plus the latest AI recommendation, if any.
router.get("/live", async (req, res) => {
  try {
    const result = await pool.query(`
      SELECT entity_id, name, latitude, longitude, metadata
      FROM map_entities WHERE entity_type = 'traffic_signal' AND metadata->>'is_demo' = 'true'`);
    const locations = result.rows
      .map((row) => ({
        id: row.entity_id,
        name: row.name,
        lat: Number(row.latitude),
        lng: Number(row.longitude),
        severity: row.metadata.traffic_level || "LOW",
        color:
          colorBySeverity[row.metadata.traffic_level] || colorBySeverity.LOW,
        reason: row.metadata.reason,
        assessment: row.metadata.assessment,
        signal: {
          signalId: row.entity_id,
          currentGreenTime: row.metadata.green_time,
          currentRedTime: row.metadata.red_time,
        },
      }))
      .sort(
        (left, right) =>
          (severityOrder[left.severity] ?? 99) -
          (severityOrder[right.severity] ?? 99),
      );
    res.json({
      city: "Mumbai",
      source: "DEMONSTRATION",
      updatedAt: new Date().toISOString(),
      locations,
    });
  } catch (error) {
    console.error("Traffic Live API Error:", error);
    res.status(500).json({ error: "Failed to fetch live traffic state." });
  }
});

// POST /api/traffic/condition - analyse a non-incident traffic condition and record a recommendation.
router.post("/condition", async (req, res) => {
  try {
    const event = await enrichSignalTiming({
      id: req.body.id || `traffic:${req.body.location || "unknown"}`,
      type: "TRAFFIC_CONDITION",
      location: req.body.location,
      severity: req.body.severity,
      metadata: req.body.metadata || {},
    });
    const trafficAgent = agentCoordinator.getAgent("TrafficAgent");
    const decision = await trafficAgent.evaluate(event);
    const record = await persistDecision(decision);
    res.status(decision ? 201 : 200).json(decisionResponse(decision, record));
  } catch (error) {
    console.error("Traffic condition evaluation error:", error);
    res.status(500).json({ error: "Unable to evaluate the traffic condition" });
  }
});

// POST /api/traffic/routes/evaluate - rank client-supplied route conditions without inventing incidents.
router.post("/routes/evaluate", async (req, res) => {
  try {
    const event = {
      id:
        req.body.id ||
        `route:${req.body.origin || "unknown"}:${req.body.destination || "unknown"}`,
      type: "ROUTE_EVALUATION",
      location: req.body.location || req.body.origin,
      severity: req.body.severity || "Moderate",
      metadata: {
        ...req.body.metadata,
        origin: req.body.origin,
        destination: req.body.destination,
        routes: req.body.routes,
        activeIncidents: req.body.activeIncidents || [],
        source: req.body.source || "DEMONSTRATION",
      },
    };
    const decision = await agentCoordinator
      .getAgent("TrafficAgent")
      .evaluate(event);
    const record = await persistDecision(decision);
    res.status(decision ? 201 : 422).json(decisionResponse(decision, record));
  } catch (error) {
    console.error("Route evaluation error:", error);
    res.status(500).json({ error: "Unable to evaluate routes" });
  }
});

// GET /api/traffic/decisions and /metrics expose durable records for operator review and aggregation.
router.get("/decisions", async (req, res) => {
  try {
    const limit = Math.min(Math.max(Number(req.query.limit) || 20, 1), 100);
    const result = await pool.query(
      "SELECT * FROM traffic_decisions ORDER BY created_at DESC LIMIT $1",
      [limit],
    );
    res.json({ decisions: result.rows });
  } catch (error) {
    console.error("Traffic decisions API error:", error);
    res.status(500).json({ error: "Failed to fetch traffic decisions" });
  }
});

router.get("/metrics", async (req, res) => {
  try {
    const result = await pool.query(`SELECT
      COUNT(*) FILTER (WHERE decision_type = 'SIGNAL_OPTIMIZATION')::int AS "signalOptimizations",
      COUNT(*) FILTER (WHERE decision_type = 'ROUTE_RECOMMENDATION')::int AS "routeRecommendations",
      COUNT(*) FILTER (WHERE payload ? 'prediction')::int AS "predictionAlerts",
      COUNT(*) FILTER (WHERE outcome = 'SUCCESS')::int AS "successfulRecommendations",
      COUNT(*) FILTER (WHERE status IN ('RECOMMENDED', 'SIMULATED'))::int AS "activeInterventions"
      FROM traffic_decisions`);
    res.json(result.rows[0]);
  } catch (error) {
    console.error("Traffic metrics API error:", error);
    res.status(500).json({ error: "Failed to fetch traffic metrics" });
  }
});

// GET /api/traffic/signals
router.get("/signals", async (req, res) => {
  try {
    const {
      page = 1,
      limit = 50,
      area,
      traffic,
      status,
      decision,
      search,
    } = req.query;
    const offset = (page - 1) * limit;

    let conditions = [];
    let values = [];
    let paramIndex = 1;

    if (area && area !== "All") {
      conditions.push(`area = $${paramIndex++}`);
      values.push(area);
    }
    if (traffic && traffic !== "All") {
      conditions.push(`traffic_level = $${paramIndex++}`);
      values.push(traffic);
    }
    if (status && status !== "All") {
      conditions.push(`status = $${paramIndex++}`);
      values.push(status);
    }
    if (decision && decision !== "All") {
      conditions.push(`ai_decision = $${paramIndex++}`);
      values.push(decision);
    }

    if (search) {
      conditions.push(
        `(signal_code ILIKE $${paramIndex} OR intersection_name ILIKE $${paramIndex} OR area ILIKE $${paramIndex})`,
      );
      values.push(`%${search}%`);
      paramIndex++;
    }

    const whereClause =
      conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";

    const countQuery = `SELECT COUNT(*) FROM traffic_signal_monitoring ${whereClause}`;
    const totalResult = await pool.query(countQuery, values);
    const total = parseInt(totalResult.rows[0].count);

    const dataQuery = `
      SELECT * FROM traffic_signal_monitoring 
      ${whereClause} 
      ORDER BY 
        CASE traffic_level
          WHEN 'CRITICAL' THEN 1 WHEN 'SEVERE' THEN 2 WHEN 'HIGH' THEN 3 WHEN 'MODERATE' THEN 4 ELSE 5
        END,
        last_observed_at DESC
      LIMIT $${paramIndex++} OFFSET $${paramIndex++}
    `;

    const signalsResult = await pool.query(dataQuery, [
      ...values,
      limit,
      offset,
    ]);

    res.json({
      signals: signalsResult.rows,
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total,
        totalPages: Math.ceil(total / limit),
      },
    });
  } catch (error) {
    console.error("API Error:", error);
    res.status(500).json({ error: "Failed to fetch signals" });
  }
});

// GET /api/traffic/signals/summary
router.get("/signals/summary", async (req, res) => {
  try {
    const query = `
      WITH signal_data AS (
        SELECT 
          m.entity_id,
          m.metadata->>'traffic_level' as level,
          d.recommendation
        FROM map_entities m
        LEFT JOIN LATERAL (
          SELECT recommendation FROM traffic_decisions 
          WHERE decision_type = 'SIGNAL_OPTIMIZATION' AND location = m.name
          ORDER BY created_at DESC LIMIT 1
        ) d ON true
        WHERE m.entity_type = 'traffic_signal'
      )
      SELECT 
        COUNT(*) AS "totalSignals",
        COUNT(*) FILTER (WHERE level IN ('HIGH', 'SEVERE', 'CRITICAL')) AS "attentionRequired",
        COUNT(*) FILTER (WHERE level = 'HIGH') AS "highCongestion",
        COUNT(*) FILTER (WHERE level IN ('SEVERE', 'CRITICAL')) AS "criticalSignals",
        COUNT(*) FILTER (WHERE recommendation IS NOT NULL) AS "optimizations"
      FROM signal_data;
    `;
    const result = await pool.query(query);
    res.json(result.rows[0]);
  } catch (error) {
    console.error("Error fetching signal summary:", error);
    res.status(500).json({ error: "Failed to fetch summary" });
  }
});

// GET /api/traffic/signals/areas
router.get("/signals/areas", async (req, res) => {
  try {
    const query = `
      WITH base_data AS (
        SELECT 
          SPLIT_PART(m.name, ' ', 1) AS area,
          m.metadata->>'traffic_level' AS level,
          d.recommendation
        FROM map_entities m
        LEFT JOIN LATERAL (
          SELECT recommendation FROM traffic_decisions 
          WHERE decision_type = 'SIGNAL_OPTIMIZATION' AND location = m.name
          ORDER BY created_at DESC LIMIT 1
        ) d ON true
        WHERE m.entity_type = 'traffic_signal'
      )
      SELECT 
        area,
        COUNT(*) AS "signalCount",
        COUNT(*) FILTER (WHERE level = 'HIGH') AS "highTraffic",
        COUNT(*) FILTER (WHERE level = 'SEVERE') AS "severeTraffic",
        COUNT(*) FILTER (WHERE level = 'CRITICAL') AS "criticalTraffic",
        COUNT(*) FILTER (WHERE recommendation IS NOT NULL) AS "optimizations"
      FROM base_data
      GROUP BY area
      ORDER BY "signalCount" DESC;
    `;
    const result = await pool.query(query);
    res.json(result.rows);
  } catch (error) {
    console.error("Error fetching area summary:", error);
    res.status(500).json({ error: "Failed to fetch area summary" });
  }
});

// GET /api/traffic/signals/activity
router.get("/signals/activity", async (req, res) => {
  try {
    const query = `
      SELECT 
        created_at AS timestamp,
        location AS signal,
        SPLIT_PART(location, ' ', 1) AS area,
        severity AS "trafficCondition",
        reason AS decision,
        payload->'prediction'->>'predictedState' AS prediction,
        recommendation,
        status
      FROM traffic_decisions 
      WHERE agent = 'TrafficAgent' AND decision_type = 'SIGNAL_OPTIMIZATION'
      ORDER BY created_at DESC 
      LIMIT 20;
    `;
    const result = await pool.query(query);
    res.json(result.rows);
  } catch (error) {
    console.error("Error fetching activity:", error);
    res.status(500).json({ error: "Failed to fetch agent activity" });
  }
});

// GET /api/traffic/signals/deployments
router.get("/signals/deployments", async (req, res) => {
  try {
    const query = `
      SELECT 
        id, 
        name, 
        type, 
        location, 
        severity, 
        description, 
        traffic_level AS "trafficLevel",
        density_level AS "densityLevel",
        flow_level AS "flowLevel",
        affected_approach AS "affectedApproach",
        source,
        is_demo AS "isDemo"
      FROM traffic_demo_scenarios 
      WHERE is_demo = true
      ORDER BY id ASC;
    `;
    const result = await pool.query(query);
    res.json(result.rows);
  } catch (error) {
    console.error("Error fetching deployments:", error);
    res.status(500).json({ error: "Failed to fetch deployments" });
  }
});

function parseGoogleDurationSeconds(value) {
  if (!value) return 0;

  const match = String(value).match(/^([\d.]+)s$/);

  return match ? Number(match[1]) : 0;
}

function googleTrafficLevel(delayMinutes, staticDurationSeconds) {
  const staticMinutes = staticDurationSeconds / 60;

  if (!staticMinutes) return "LOW";

  const delayRatio = delayMinutes / staticMinutes;

  if (delayRatio >= 0.6) return "SEVERE";
  if (delayRatio >= 0.35) return "HIGH";
  if (delayRatio >= 0.15) return "MODERATE";

  return "LOW";
}

async function computeGoogleRoute({
  origin,
  destination,
  departureTime = null,
  trafficModel = "BEST_GUESS",
}) {
  const apiKey =
    process.env.GOOGLE_MAPS_API_KEY ||
    process.env.GOOGLE_ROUTES_API_KEY;

  if (!apiKey) {
    const error = new Error(
      "Google Routes API key is not configured on the backend."
    );

    error.statusCode = 500;

    throw error;
  }

  const isPredictive = Boolean(departureTime);

  const requestBody = {
    origin: {
      address: origin,
    },

    destination: {
      address: destination,
    },

    travelMode: "DRIVE",

    routingPreference: isPredictive
      ? "TRAFFIC_AWARE_OPTIMAL"
      : "TRAFFIC_AWARE",

    computeAlternativeRoutes: true,

    extraComputations: [
      "TRAFFIC_ON_POLYLINE",
    ],

    ...(isPredictive
      ? {
          departureTime,
          trafficModel,
        }
      : {}),
  };

  const response = await fetch(
    "https://routes.googleapis.com/directions/v2:computeRoutes",
    {
      method: "POST",

      headers: {
        "Content-Type": "application/json",

        "X-Goog-Api-Key": apiKey,

        "X-Goog-FieldMask":
          "routes.routeLabels," +
          "routes.distanceMeters," +
          "routes.duration," +
          "routes.staticDuration," +
          "routes.polyline.encodedPolyline," +
          "routes.travelAdvisory.speedReadingIntervals",
      },

      body: JSON.stringify(requestBody),
    }
  );

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const error = new Error(
      data?.error?.message ||
        "Google Routes API request failed."
    );

    error.statusCode = response.status;

    throw error;
  }

  return (data.routes || []).map((route, index) => {
    const durationSeconds =
      parseGoogleDurationSeconds(route.duration);

    const staticDurationSeconds =
      parseGoogleDurationSeconds(
        route.staticDuration
      ) || durationSeconds;

    const delayMinutes = Math.max(
      0,
      Math.round(
        (durationSeconds -
          staticDurationSeconds) /
          60
      )
    );

    return {
      id: index,

      name: `Route ${String.fromCharCode(
        65 + index
      )}`,

      distanceKm: Number(
        (
          (route.distanceMeters || 0) /
          1000
        ).toFixed(1)
      ),

      durationSeconds,

      staticDurationSeconds,

      delayMinutes,

      trafficLevel: googleTrafficLevel(
        delayMinutes,
        staticDurationSeconds
      ),

      polyline:
        route.polyline?.encodedPolyline ||
        null,

      speedReadingIntervals:
        route.travelAdvisory
          ?.speedReadingIntervals || [],

      source: "GOOGLE_ROUTES_API",

      isPredicted: isPredictive,
    };
  });
}

// POST /api/traffic/google/predict
router.post("/google/predict", async (req, res) => {
  try {
    const {
      origin,
      destination,
      departureTime,
      trafficModel = "BEST_GUESS",
    } = req.body || {};

    if (!origin || !destination || !departureTime) {
      return res.status(400).json({
        error:
          "origin, destination and a future departureTime are required.",
      });
    }

    const departure = new Date(departureTime);

    if (Number.isNaN(departure.getTime())) {
      return res.status(400).json({
        error:
          "departureTime must be a valid ISO timestamp.",
      });
    }

    if (departure.getTime() <= Date.now()) {
      return res.status(400).json({
        error:
          "departureTime must be in the future for Google traffic prediction.",
      });
    }

    const routes = await computeGoogleRoute({
      origin,
      destination,
      departureTime: departure.toISOString(),
      trafficModel,
    });

    res.json({
      source: "GOOGLE_TRAFFIC_MODEL",
      departureTime: departure.toISOString(),
      trafficModel,
      routes,
    });
  } catch (error) {
    console.error(
      "Google traffic prediction error:",
      error
    );

    res.status(error.statusCode || 500).json({
      error:
        error.message ||
        "Unable to calculate Google traffic prediction.",
    });
  }
});

// GET /api/traffic/history
router.get("/history", async (req, res) => {
  try {
    const {
      date,
      time = "18:30",
    } = req.query;

    if (!date) {
      return res.status(400).json({
        error:
          "date is required in YYYY-MM-DD format.",
      });
    }

    const result = await pool.query(
      `
      SELECT
        id,
        corridor_id,
        corridor_name,
        observed_at,
        latitude,
        longitude,
        encoded_polyline,
        speed_category,
        average_speed_kmh,
        delay_minutes,
        traffic_level,
        source
      FROM traffic_history
      WHERE observed_at >= (
        $1::timestamp
        AT TIME ZONE 'Asia/Kolkata'
      ) - INTERVAL '5 minutes'

        AND observed_at <= (
          $1::timestamp
          AT TIME ZONE 'Asia/Kolkata'
        ) + INTERVAL '5 minutes'

      ORDER BY
        observed_at ASC,
        corridor_name ASC
      `,
      [`${date} ${time}:00`]
    );

    res.json({
      source: "ARCHIVED_GOOGLE_TRAFFIC",

      requestedDate: date,

      requestedTime: time,

      locations: result.rows,

      count: result.rows.length,

      message:
        result.rows.length > 0
          ? "Archived Google traffic observations loaded."
          : "No archived Google traffic observations exist for this timestamp.",
    });
  } catch (error) {
    console.error(
      "Traffic history API error:",
      error
    );

    res.status(500).json({
      error:
        "Failed to fetch traffic history. Run the traffic_history migration first.",
    });
  }
});
// POST /api/traffic/history/capture
router.post("/history/capture", async (req, res) => {
  try {
    const {
      corridorId,
      corridorName,
      origin,
      destination,
      latitude = null,
      longitude = null,
    } = req.body || {};

    if (
      !corridorId ||
      !corridorName ||
      !origin ||
      !destination
    ) {
      return res.status(400).json({
        error:
          "corridorId, corridorName, origin and destination are required.",
      });
    }

    const routes = await computeGoogleRoute({
      origin,
      destination,
    });

    const route = routes[0];

    if (!route) {
      return res.status(404).json({
        error:
          "Google returned no route for this corridor.",
      });
    }

    const averageSpeed =
      route.durationSeconds > 0 &&
      route.distanceKm > 0
        ? Number(
            (
              route.distanceKm /
              (route.durationSeconds / 3600)
            ).toFixed(1)
          )
        : null;

    const result = await pool.query(
      `
      INSERT INTO traffic_history (
        corridor_id,
        corridor_name,
        observed_at,
        latitude,
        longitude,
        encoded_polyline,
        speed_category,
        average_speed_kmh,
        delay_minutes,
        traffic_level,
        source,
        metadata
      )
      VALUES (
        $1,
        $2,
        NOW(),
        $3,
        $4,
        $5,
        $6,
        $7,
        $8,
        $9,
        $10,
        $11::jsonb
      )
      RETURNING *
      `,
      [
        String(corridorId),

        corridorName,

        latitude === null
          ? null
          : Number(latitude),

        longitude === null
          ? null
          : Number(longitude),

        route.polyline,

        route.speedReadingIntervals?.length
          ? "MIXED"
          : route.trafficLevel,

        averageSpeed,

        route.delayMinutes,

        route.trafficLevel,

        "GOOGLE_ROUTES_API",

        JSON.stringify({
          origin,
          destination,
          routeName: route.name,
          speedReadingIntervals:
            route.speedReadingIntervals || [],
        }),
      ]
    );

    res.status(201).json({
      source: "GOOGLE_ROUTES_API",

      observation: result.rows[0],
    });
  } catch (error) {
    console.error(
      "Traffic history capture error:",
      error
    );

    res.status(error.statusCode || 500).json({
      error:
        error.message ||
        "Failed to capture Google traffic observation.",
    });
  }
});
module.exports = router;