const express = require("express");
const router = express.Router();
const db = require("../../config/db"); 


// ============================================================
// GET /api/police/map-data
// Fetch incidents + crimes for Police Operations Map
// ============================================================
router.get("/map-data", async (req, res) => {
  try {
    // --------------------------------------------------------
    // 1. ACTIVE INCIDENTS
    // --------------------------------------------------------
    const incidentsQuery = `
      SELECT
        incident_id,
        incident_type,
        COALESCE(
          timeline_events->>'location',
          'Unknown Location'
        ) AS location,
        latitude,
        longitude,
        severity,
        status,
        created_at AS reported_at,
        assigned_departments,
        timeline_events
      FROM incidents
      WHERE latitude IS NOT NULL
        AND longitude IS NOT NULL
        AND status ILIKE 'active'
      ORDER BY created_at DESC;
    `;

    // --------------------------------------------------------
    // 2. CRIME RECORDS
    // --------------------------------------------------------
    const crimesQuery = `
      SELECT
        id,
        category,
        location,
        latitude,
        longitude,
        severity,
        status,
        reported_at
      FROM crimes
      WHERE latitude IS NOT NULL
        AND longitude IS NOT NULL
      ORDER BY reported_at DESC;
    `;

    const [incidentsResult, crimesResult] = await Promise.all([
      db.query(incidentsQuery),
      db.query(crimesQuery),
    ]);

    // --------------------------------------------------------
    // 3. RETURN MAP DATA
    // --------------------------------------------------------
    res.json({
      success: true,
      incidents: incidentsResult.rows,
      crimes: crimesResult.rows,
      counts: {
        incidents: incidentsResult.rows.length,
        crimes: crimesResult.rows.length,
      },
    });

  } catch (error) {
    console.error(
      "[POLICE MAP] Failed to fetch map data:",
      error
    );

    res.status(500).json({
      success: false,
      error: "Failed to fetch police map data",
    });
  }
});
// GET /api/police/overview
router.get("/overview", async (req, res) => {
  try {
    const query = `
      WITH active_incidents AS (
        SELECT 
          COUNT(*) as total_active,
          COUNT(*) FILTER (WHERE severity ILIKE 'CRITICAL' OR severity ILIKE 'HIGH') as high_priority
        FROM incidents 
        WHERE status ILIKE 'active'
      ),
      police_stations AS (
        SELECT 
          COUNT(*) as total_stations,
          COALESCE(SUM((metadata->>'patrolsAvailable')::int), 0) as available_patrols
        FROM map_entities 
        WHERE entity_type = 'police_station'
      )
      SELECT 
        (SELECT total_active FROM active_incidents) as "activeIncidents",
        (SELECT high_priority FROM active_incidents) as "criticalIncidents",
        (SELECT available_patrols FROM police_stations) as "unitsAvailable",
        (SELECT total_stations * 3 FROM police_stations) as "unitsDeployed";
    `;
    const result = await db.query(query);
    res.json(result.rows[0] || { activeIncidents: 0, criticalIncidents: 0, unitsAvailable: 0, unitsDeployed: 0 });
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch police metrics" });
  }
});

// GET /api/police/incidents
router.get("/incidents", async (req, res) => {
  try {
    const query = `
      SELECT 
        incident_id, incident_type, 
        COALESCE(timeline_events->>'location', 'Target Perimeter') as location, 
        latitude, longitude, severity, status, created_at as reported_at, assigned_departments
      FROM incidents
      WHERE status ILIKE 'active'
      ORDER BY created_at DESC LIMIT 10;
    `;
    const result = await db.query(query);
    res.json(result.rows);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch incidents" });
  }
});

// GET /api/police/crimes - NEW ENDPOINT
router.get("/crimes", async (req, res) => {
  try {
    const query = `
      SELECT id, category, location, latitude, longitude, severity, status, reported_at as time
      FROM crimes
      ORDER BY reported_at DESC;
    `;
    const result = await db.query(query);
    res.json(result.rows);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch crimes" });
  }
});

// GET /api/police/agent/recommendation
router.get("/agent/recommendation", async (req, res) => {
  try {
    const query = `
      SELECT agent, decision_type, location, severity, recommendation, reason, status, source, created_at
      FROM traffic_decisions 
      WHERE agent IN ('PoliceAgent', 'TrafficAgent')
      ORDER BY created_at DESC LIMIT 1;
    `;
    const result = await db.query(query);
    res.json(result.rows[0] || null);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch agent recommendation" });
  }
});
// ==========================================================
// GET /api/police/stations
// Fetch real police stations from map_entities
// ==========================================================

router.get("/stations", async (req, res) => {
  try {
    const query = `
      SELECT
        entity_id,
        name,
        entity_type,
        latitude,
        longitude,
        metadata,
        created_at
      FROM map_entities
      WHERE LOWER(TRIM(entity_type)) = 'police_station'
        AND latitude IS NOT NULL
        AND longitude IS NOT NULL
      ORDER BY name ASC;
    `;

    const result = await db.query(query);

    console.log(
      `[POLICE STATIONS] Found ${result.rows.length} stations`
    );

    console.log(
      "[POLICE STATIONS] Data:",
      result.rows
    );

    res.json({
      success: true,
      count: result.rows.length,
      stations: result.rows,
    });

  } catch (error) {
    console.error(
      "[POLICE STATIONS] Database error:",
      error
    );

    res.status(500).json({
      success: false,
      error: "Failed to fetch police stations",
      details: error.message,
    });
  }
});

// ============================================================
// PASTE THESE ROUTES INTO policeRoutes.js
// (after the existing GET "/incidents" route, before module.exports).
//
// The existing GET /incidents (active only, LIMIT 10) is left untouched so
// the Police store keeps working. Order matters: "/incidents/all" must be
// declared before "/incidents/:id".
// ============================================================

const INCIDENT_COLUMNS = `
  i.*,
  COALESCE(i.timeline_events->>'location', 'Unknown Location') AS location,
  i.created_at AS reported_at
`;

// Great-circle distance in km (no PostGIS needed).
const distanceSql = (latCol, lngCol, latParam, lngParam) => `
  (6371 * acos(LEAST(1,
    cos(radians(${latParam}::float8)) * cos(radians(${latCol}::float8)) *
    cos(radians(${lngCol}::float8) - radians(${lngParam}::float8)) +
    sin(radians(${latParam}::float8)) * sin(radians(${latCol}::float8))
  )))
`;

const VALID_STATUSES = ["ACTIVE", "PENDING", "RESOLVED", "CLOSED"];
const VALID_EVENTS = [
  "INCIDENT_UPDATE",
  "UNIT_DISPATCHED",
  "UNIT_ARRIVED",
  "COORDINATION_REQUEST",
];

// Optional realtime hook-up: if your server exposes socket.io via
// app.set("io", io), connected clients are told to refresh. Rename the event
// to match what your Police store already listens for. (The frontend also
// refetches right after every action, so this is an enhancement only.)
const notifyIncidentChange = (req, incidentId) => {
  try {
    req.app.get("io")?.emit("incident:updated", { incident_id: incidentId });
  } catch (err) {
    console.error("[POLICE INCIDENTS] socket notify failed:", err.message);
  }
};

// ------------------------------------------------------------
// GET /api/police/incidents/all  -> full incident history
// ------------------------------------------------------------
router.get("/incidents/all", async (req, res) => {
  try {
    const result = await db.query(`
      SELECT ${INCIDENT_COLUMNS}
      FROM incidents i
      ORDER BY i.created_at DESC
      LIMIT 1000;
    `);
    res.json({ success: true, count: result.rows.length, incidents: result.rows });
  } catch (error) {
    console.error("[POLICE INCIDENTS] list failed:", error);
    res.status(500).json({ success: false, error: "Failed to fetch incidents" });
  }
});

// ------------------------------------------------------------
// GET /api/police/incidents/:id
// One incident + related incidents + nearest stations + nearby crimes
// ------------------------------------------------------------
router.get("/incidents/:id", async (req, res) => {
  try {
    const found = await db.query(
      `SELECT ${INCIDENT_COLUMNS} FROM incidents i WHERE i.incident_id::text = $1 LIMIT 1;`,
      [req.params.id],
    );

    if (found.rows.length === 0) {
      return res.status(404).json({ success: false, error: "Incident not found" });
    }

    const incident = found.rows[0];
    const lat = incident.latitude !== null ? Number(incident.latitude) : null;
    const lng = incident.longitude !== null ? Number(incident.longitude) : null;
    const hasCoords = Number.isFinite(lat) && Number.isFinite(lng);

    const relatedQuery = db.query(
      `
      SELECT * FROM (
        SELECT
          ${INCIDENT_COLUMNS},
          CASE WHEN i.latitude IS NULL OR i.longitude IS NULL OR $4::float8 IS NULL THEN NULL
               ELSE ${distanceSql("i.latitude", "i.longitude", "$4", "$5")} END AS distance_km
        FROM incidents i
        WHERE i.incident_id::text <> $1
      ) t
      WHERE t.incident_type = $2
         OR t.location = $3
         OR (t.distance_km IS NOT NULL AND t.distance_km <= 2)
      ORDER BY t.reported_at DESC
      LIMIT 8;
      `,
      [incident.incident_id, incident.incident_type, incident.location, hasCoords ? lat : null, hasCoords ? lng : null],
    );

    const stationsQuery = hasCoords
      ? db.query(
          `
          SELECT entity_id, name, latitude, longitude, metadata,
                 ${distanceSql("latitude", "longitude", "$1", "$2")} AS distance_km
          FROM map_entities
          WHERE LOWER(TRIM(entity_type)) = 'police_station'
            AND latitude IS NOT NULL AND longitude IS NOT NULL
          ORDER BY distance_km ASC
          LIMIT 5;
          `,
          [lat, lng],
        )
      : Promise.resolve({ rows: [] });

    const crimesQuery = hasCoords
      ? db.query(
          `
          SELECT * FROM (
            SELECT id, category, location, latitude, longitude, severity, status, reported_at,
                   ${distanceSql("latitude", "longitude", "$1", "$2")} AS distance_km
            FROM crimes
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
          ) c
          WHERE c.distance_km <= 3
          ORDER BY c.reported_at DESC
          LIMIT 20;
          `,
          [lat, lng],
        )
      : Promise.resolve({ rows: [] });

    const [related, stations, crimes] = await Promise.all([relatedQuery, stationsQuery, crimesQuery]);

    res.json({
      success: true,
      incident,
      related: related.rows,
      nearbyStations: stations.rows,
      nearbyCrimes: crimes.rows,
    });
  } catch (error) {
    console.error("[POLICE INCIDENTS] detail failed:", error);
    res.status(500).json({ success: false, error: "Failed to fetch incident" });
  }
});

// ------------------------------------------------------------
// PATCH /api/police/incidents/:id/status   { status }
// Updates status and appends a STATUS_CHANGED timeline event.
// ------------------------------------------------------------
router.patch("/incidents/:id/status", async (req, res) => {
  try {
    const next = String(req.body?.status || "").trim().toUpperCase();
    if (!VALID_STATUSES.includes(next)) {
      return res.status(400).json({ success: false, error: `Status must be one of: ${VALID_STATUSES.join(", ")}` });
    }

    const current = await db.query(
      "SELECT incident_id, status FROM incidents WHERE incident_id::text = $1 LIMIT 1;",
      [req.params.id],
    );
    if (current.rows.length === 0) {
      return res.status(404).json({ success: false, error: "Incident not found" });
    }

    const previous = String(current.rows[0].status || "");
    // Preserve the casing convention already used in the table.
    const stored = previous && previous === previous.toLowerCase() ? next.toLowerCase() : next;

    const event = {
      time: new Date().toISOString(),
      event: "STATUS_CHANGED",
      description: `Status changed from ${previous.toUpperCase() || "UNKNOWN"} to ${next}`,
      status: next,
    };

    await db.query(
      `
      UPDATE incidents
      SET status = $2,
          timeline_events = jsonb_set(
            COALESCE(timeline_events::jsonb, '{}'::jsonb),
            '{timeline}',
            COALESCE(timeline_events::jsonb->'timeline', '[]'::jsonb) || $3::jsonb,
            true
          )
      WHERE incident_id::text = $1;
      `,
      [req.params.id, stored, JSON.stringify([event])],
    );

    notifyIncidentChange(req, current.rows[0].incident_id);
    res.json({ success: true, status: stored });
  } catch (error) {
    console.error("[POLICE INCIDENTS] status update failed:", error);
    res.status(500).json({ success: false, error: "Failed to update incident status" });
  }
});

// ------------------------------------------------------------
// POST /api/police/incidents/:id/timeline   { event, description }
// ------------------------------------------------------------
router.post("/incidents/:id/timeline", async (req, res) => {
  try {
    const eventType = String(req.body?.event || "").trim().toUpperCase();
    const description = String(req.body?.description || "").trim();

    if (!VALID_EVENTS.includes(eventType)) {
      return res.status(400).json({ success: false, error: `Event must be one of: ${VALID_EVENTS.join(", ")}` });
    }
    if (!description || description.length > 500) {
      return res.status(400).json({ success: false, error: "Description is required (max 500 characters)" });
    }

    const event = { time: new Date().toISOString(), event: eventType, description };

    const result = await db.query(
      `
      UPDATE incidents
      SET timeline_events = jsonb_set(
            COALESCE(timeline_events::jsonb, '{}'::jsonb),
            '{timeline}',
            COALESCE(timeline_events::jsonb->'timeline', '[]'::jsonb) || $2::jsonb,
            true
          )
      WHERE incident_id::text = $1
      RETURNING incident_id;
      `,
      [req.params.id, JSON.stringify([event])],
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ success: false, error: "Incident not found" });
    }

    notifyIncidentChange(req, result.rows[0].incident_id);
    res.json({ success: true, event });
  } catch (error) {
    console.error("[POLICE INCIDENTS] timeline add failed:", error);
    res.status(500).json({ success: false, error: "Failed to add timeline event" });
  }
});


module.exports = router;