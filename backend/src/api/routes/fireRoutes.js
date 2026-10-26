// backend/src/routes/fireRoutes.js
// Merged: former fireRoutes.js (incidents) + fireDispatchRoutes.js (dispatch).
// Mount once:  app.use('/api/fire', require('./routes/fireRoutes'));
//
//   /api/fire/incidents/summary    incident KPIs
//   /api/fire/incidents            incident list (filters, limit)
//   /api/fire/incidents/:id        one incident, by incident_number
//   /api/fire/dispatch/summary     dispatch KPIs + resource utilization
//   /api/fire/dispatch/stations    station workload
//   /api/fire/dispatch             prioritized dispatch queue
//   /api/fire/stations, /api/fire  legacy aliases of the two dispatch routes above
const express = require('express');
const router = express.Router();
const pool = require('../../config/db');

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// One definition of "finished", so every endpoint agrees on what counts as active.
const CLOSED_STATUSES = `('RESOLVED', 'CLOSED')`;

const STATUS_ORDER = `
  CASE status
    WHEN 'REPORTED' THEN 1 WHEN 'ACKNOWLEDGED' THEN 2 WHEN 'DISPATCHED' THEN 3
    WHEN 'EN_ROUTE' THEN 4 WHEN 'ON_SCENE' THEN 5 WHEN 'CONTAINED' THEN 6
    ELSE 7
  END`;

const SEVERITY_ORDER = `
  CASE severity
    WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MODERATE' THEN 3 WHEN 'LOW' THEN 4
    ELSE 5
  END`;

const PRIORITY_ORDER = `
  CASE priority
    WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 WHEN 'LOW' THEN 4
    ELSE 5
  END`;

// The frontend reads `created_at`; the column is `reported_at`. Expose both.
const INCIDENT_COLUMNS = `*, reported_at AS created_at`;

const DEFAULT_LIMIT = 500;
const MAX_LIMIT = 1000;

// Escape LIKE wildcards so searching for "50%" or "a_b" matches literally.
const escapeLike = (s) => String(s).replace(/[\\%_]/g, '\\$&');

const isSet = (v) => v !== undefined && v !== null && v !== '' && v !== 'ALL';

/**
 * Builds the WHERE clause shared by the list endpoints.
 * Every value goes through a bind parameter; nothing from the request
 * is concatenated into SQL.
 */
function buildFilters(q, { exactFields = [], searchFields = [], pendingGroup = false } = {}) {
  const clauses = [];
  const values = [];

  // Dispatch queue: "REPORTED" means the whole "Dispatch Pending" group.
  const groupedStatus = pendingGroup && q.status === 'REPORTED';
  if (groupedStatus) clauses.push(`status IN ('REPORTED', 'ACKNOWLEDGED')`);

  exactFields.forEach((field) => {
    if (groupedStatus && field === 'status') return;
    if (isSet(q[field])) {
      values.push(q[field]);
      clauses.push(`${field} = $${values.length}`);
    }
  });

  if (q.active_only === 'true') {
    clauses.push(`status NOT IN ${CLOSED_STATUSES}`);
  }

  if (isSet(q.search) && searchFields.length > 0) {
    values.push(`%${escapeLike(String(q.search).trim())}%`);
    const p = `$${values.length}`;
    clauses.push(`(${searchFields.map((f) => `${f} ILIKE ${p}`).join(' OR ')})`);
  }

  return {
    where: clauses.length ? `WHERE ${clauses.join(' AND ')}` : '',
    values,
  };
}

function parseLimit(raw) {
  const n = parseInt(raw, 10);
  if (!Number.isFinite(n) || n <= 0) return DEFAULT_LIMIT;
  return Math.min(n, MAX_LIMIT);
}

const fail = (res, tag, message, error) => {
  console.error(`[FIRE API] ${tag}:`, error);
  res.status(500).json({ error: message });
};

// ---------------------------------------------------------------------------
// GET /api/fire/incidents/summary
// ---------------------------------------------------------------------------
router.get('/incidents/summary', async (req, res) => {
  try {
    const query = `
      SELECT
        COUNT(*) FILTER (WHERE status NOT IN ${CLOSED_STATUSES})::int AS "activeIncidents",
        COUNT(*) FILTER (WHERE severity = 'CRITICAL' AND status NOT IN ${CLOSED_STATUSES})::int AS "critical",
        COUNT(*) FILTER (WHERE priority = 'HIGH' AND status NOT IN ${CLOSED_STATUSES})::int AS "highPriority",
        COUNT(*) FILTER (WHERE status = 'ON_SCENE')::int AS "onScene",
        COUNT(*) FILTER (WHERE status = 'CONTAINED')::int AS "contained",
        ROUND(AVG(response_time_minutes))::int AS "avgResponseTime"
      FROM fire_incidents
    `;
    const result = await pool.query(query);
    res.json(result.rows[0]);
  } catch (error) {
    fail(res, 'Error fetching summary metrics', 'Failed to fetch fire metrics', error);
  }
});

// ---------------------------------------------------------------------------
// GET /api/fire/incidents
// Query: severity, status, incident_type, ward, search, active_only, limit
// ---------------------------------------------------------------------------
router.get('/incidents', async (req, res) => {
  try {
    const { where, values } = buildFilters(req.query, {
      exactFields: ['severity', 'status', 'incident_type', 'ward'],
      searchFields: ['incident_number', 'location', 'address'],
    });

    values.push(parseLimit(req.query.limit));
    const query = `
      SELECT ${INCIDENT_COLUMNS}
      FROM fire_incidents
      ${where}
      ORDER BY ${STATUS_ORDER}, reported_at DESC
      LIMIT $${values.length}
    `;

    const result = await pool.query(query, values);
    res.json(result.rows);
  } catch (error) {
    fail(res, 'Error fetching incidents', 'Failed to fetch fire incidents', error);
  }
});

// ---------------------------------------------------------------------------
// GET /api/fire/incidents/:id   (looked up by incident_number)
// ---------------------------------------------------------------------------
router.get('/incidents/:id', async (req, res) => {
  try {
    const result = await pool.query(
      `SELECT ${INCIDENT_COLUMNS} FROM fire_incidents WHERE incident_number = $1`,
      [req.params.id]
    );
    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Incident not found' });
    }
    res.json(result.rows[0]);
  } catch (error) {
    fail(res, 'Error fetching incident details', 'Failed to fetch incident details', error);
  }
});

// ---------------------------------------------------------------------------
// Dispatch: summary, station workload, queue
// ---------------------------------------------------------------------------

// GET /api/fire/dispatch/summary - operational KPIs and resource utilization
const dispatchSummary = async (req, res) => {
  try {
    const query = `
      SELECT
        COUNT(*) FILTER (WHERE status NOT IN ${CLOSED_STATUSES})::int AS active_incidents,
        COUNT(*) FILTER (WHERE status IN ('REPORTED', 'ACKNOWLEDGED'))::int AS dispatch_pending,
        COUNT(*) FILTER (WHERE status = 'DISPATCHED')::int AS dispatched_incidents,
        COUNT(*) FILTER (WHERE status = 'EN_ROUTE')::int AS en_route,
        COUNT(*) FILTER (WHERE status = 'ON_SCENE')::int AS on_scene,
        COUNT(*) FILTER (WHERE status = 'CONTAINED')::int AS contained,
        ROUND(AVG(response_time_minutes))::int AS avg_response_time,
        COALESCE(SUM(fire_tenders_required), 0)::int AS total_tenders_required,
        COALESCE(SUM(fire_tenders_dispatched), 0)::int AS total_tenders_dispatched,
        COALESCE(SUM(rescue_units_dispatched), 0)::int AS total_rescue_dispatched,
        COALESCE(SUM(water_tankers_dispatched), 0)::int AS total_tankers_dispatched
      FROM fire_incidents
      WHERE status NOT IN ${CLOSED_STATUSES} OR reported_at >= NOW() - INTERVAL '24 HOURS'
    `;
    const result = await pool.query(query);
    res.json(result.rows[0]);
  } catch (error) {
    fail(res, 'Error fetching dispatch summary', 'Failed to fetch dispatch summary.', error);
  }
};

// GET /api/fire/dispatch/stations - station workload
const dispatchStations = async (req, res) => {
  try {
    const query = `
      SELECT
        station_name,
        COUNT(*)::int AS active_incidents,
        COALESCE(SUM(
          COALESCE(fire_tenders_dispatched, 0) +
          COALESCE(rescue_units_dispatched, 0) +
          COALESCE(water_tankers_dispatched, 0)
        ), 0)::int AS dispatched_units,
        COUNT(*) FILTER (WHERE status = 'ON_SCENE')::int AS on_scene_incidents,
        ROUND(AVG(response_time_minutes))::int AS avg_response
      FROM fire_incidents
      WHERE status NOT IN ${CLOSED_STATUSES} AND station_name IS NOT NULL
      GROUP BY station_name
      ORDER BY active_incidents DESC, station_name
    `;
    const result = await pool.query(query);
    res.json(result.rows);
  } catch (error) {
    fail(res, 'Error fetching station workload', 'Failed to fetch station workload.', error);
  }
};

// GET /api/fire/dispatch - active dispatch queue
// Query: status, severity, incident_type, search, active_only, limit
// Order: severity -> priority -> status -> newest
const dispatchQueue = async (req, res) => {
  try {
    const { where, values } = buildFilters(req.query, {
      exactFields: ['status', 'severity', 'incident_type'],
      searchFields: ['incident_number', 'location', 'address'],
      pendingGroup: true,
    });

    values.push(parseLimit(req.query.limit));
    const query = `
      SELECT ${INCIDENT_COLUMNS}
      FROM fire_incidents
      ${where}
      ORDER BY ${SEVERITY_ORDER}, ${PRIORITY_ORDER}, ${STATUS_ORDER}, reported_at DESC
      LIMIT $${values.length}
    `;

    const result = await pool.query(query, values);
    res.json(result.rows);
  } catch (error) {
    fail(res, 'Error fetching dispatch queue', 'Failed to fetch dispatch queue.', error);
  }
};

// GET /api/fire/stations/summary
router.get('/stations/summary', async (req, res) => {
  try {
    const query = `
      WITH station_metrics AS (
        SELECT 
          m.entity_id,
          m.name,
          COUNT(i.incident_number) FILTER (WHERE i.status NOT IN ${CLOSED_STATUSES}) AS active_count
        FROM map_entities m
        LEFT JOIN fire_incidents i ON i.station_name = m.name
        WHERE m.entity_type = 'fire_station' AND m.layer_group = 'fire'
        GROUP BY m.entity_id, m.name
      )
      SELECT 
        COUNT(*)::int AS "totalStations",
        COUNT(*) FILTER (WHERE active_count = 0)::int AS "operationalStations",
        COUNT(*) FILTER (WHERE active_count > 0 AND active_count <= 2)::int AS "busyStations",
        COUNT(*) FILTER (WHERE active_count > 2)::int AS "highLoadStations",
        SUM(active_count)::int AS "activeIncidents",
        (SELECT COUNT(*) FROM fire_incidents WHERE DATE(reported_at) = CURRENT_DATE)::int AS "todayIncidents",
        (SELECT ROUND(AVG(response_time_minutes)) FROM fire_incidents WHERE response_time_minutes IS NOT NULL)::int AS "averageResponseTime"
      FROM station_metrics
    `;
    const result = await pool.query(query);
    res.json(result.rows[0]);
  } catch (error) {
    fail(res, 'Error fetching station summary', 'Failed to calculate station summary metrics.', error);
  }
});

// GET /api/fire/stations/resources
router.get('/stations/resources', async (req, res) => {
  try {
    const query = `
      SELECT 
        SUM((metadata->>'engines')::int)::int AS "totalFireEngines",
        (SELECT COALESCE(SUM(fire_tenders_dispatched), 0)::int FROM fire_incidents WHERE status NOT IN ${CLOSED_STATUSES}) AS "fireTendersDispatched",
        (SELECT COALESCE(SUM(rescue_units_dispatched), 0)::int FROM fire_incidents WHERE status NOT IN ${CLOSED_STATUSES}) AS "rescueUnitsDispatched",
        (SELECT COALESCE(SUM(water_tankers_dispatched), 0)::int FROM fire_incidents WHERE status NOT IN ${CLOSED_STATUSES}) AS "waterTankersDispatched"
      FROM map_entities
      WHERE entity_type = 'fire_station' AND layer_group = 'fire'
    `;
    const result = await pool.query(query);
    res.json(result.rows[0]);
  } catch (error) {
    fail(res, 'Error fetching station resources', 'Failed to calculate resource readiness.', error);
  }
});

// GET /api/fire/stations/workload
// Safely reuses the existing dispatch logic to prevent SQL duplication
router.get('/stations/workload', dispatchStations);

// GET /api/fire/stations/:id/incidents
router.get('/stations/:id/incidents', async (req, res) => {
  try {
    const query = `
      SELECT i.incident_number, i.incident_type, i.severity, i.priority, i.status, i.location, i.address, 
             i.reported_at, i.response_time_minutes, i.fire_tenders_required, i.fire_tenders_dispatched, 
             i.rescue_units_dispatched, i.water_tankers_dispatched
      FROM fire_incidents i
      JOIN map_entities m ON m.name = i.station_name
      WHERE m.entity_id = $1 AND m.entity_type = 'fire_station'
      ORDER BY CASE WHEN i.status NOT IN ${CLOSED_STATUSES} THEN 0 ELSE 1 END, i.reported_at DESC
    `;
    const result = await pool.query(query, [req.params.id]);
    res.json(result.rows);
  } catch (error) {
    fail(res, 'Error fetching station incidents', 'Failed to fetch assigned incidents.', error);
  }
});

// GET /api/fire/stations/:id
router.get('/stations/:id', async (req, res) => {
  try {
    const query = `
      SELECT 
        m.entity_id, m.name, m.latitude, m.longitude, m.metadata, m.layer_group, m.created_at,
        COUNT(i.incident_number) FILTER (WHERE i.status NOT IN ${CLOSED_STATUSES})::int AS active_incidents,
        COUNT(i.incident_number) FILTER (WHERE i.status = 'DISPATCHED')::int AS dispatched_incidents,
        COUNT(i.incident_number) FILTER (WHERE i.status = 'EN_ROUTE')::int AS en_route_incidents,
        COUNT(i.incident_number) FILTER (WHERE i.status = 'ON_SCENE')::int AS on_scene_incidents,
        COUNT(i.incident_number) FILTER (WHERE i.status = 'CONTAINED')::int AS contained_incidents,
        COUNT(i.incident_number) FILTER (WHERE DATE(i.reported_at) = CURRENT_DATE)::int AS today_incidents,
        ROUND(AVG(i.response_time_minutes))::int AS avg_response_time
      FROM map_entities m
      LEFT JOIN fire_incidents i ON i.station_name = m.name
      WHERE m.entity_id = $1 AND m.entity_type = 'fire_station'
      GROUP BY m.entity_id
    `;
    const result = await pool.query(query, [req.params.id]);
    if (result.rows.length === 0) return res.status(404).json({ error: 'Station not found' });
    res.json(result.rows[0]);
  } catch (error) {
    fail(res, 'Error fetching station details', 'Failed to fetch station details.', error);
  }
});

// OVERRIDE Legacy GET /api/fire/stations
router.get('/stations', async (req, res) => {
  try {
    const query = `
      SELECT 
        m.entity_id, m.name, m.latitude, m.longitude, m.metadata, m.layer_group, m.created_at,
        COUNT(i.incident_number) FILTER (WHERE i.status NOT IN ${CLOSED_STATUSES})::int AS active_incidents,
        ROUND(AVG(i.response_time_minutes))::int AS avg_response_time,
        COALESCE(SUM(i.fire_tenders_dispatched) FILTER (WHERE i.status NOT IN ${CLOSED_STATUSES}), 0)::int AS dispatched_units,
        COUNT(i.incident_number) FILTER (WHERE i.status = 'ON_SCENE')::int AS on_scene_incidents
      FROM map_entities m
      LEFT JOIN fire_incidents i ON i.station_name = m.name
      WHERE m.entity_type = 'fire_station' AND m.layer_group = 'fire'
      GROUP BY m.entity_id
      ORDER BY active_incidents DESC, m.name ASC
    `;
    const result = await pool.query(query);
    res.json(result.rows);
  } catch (error) {
    fail(res, 'Error fetching stations list', 'Failed to fetch station network.', error);
  }
});

router.get('/dispatch/summary', dispatchSummary);
router.get('/dispatch/stations', dispatchStations);
router.get('/dispatch', dispatchQueue);

// Legacy aliases: the old fireRoutes.js served these at /stations and /.
router.get('/stations', dispatchStations);
router.get('/', dispatchQueue);

module.exports = router;