// backend/src/routes/fireDispatchRoutes.js
const express = require('express');
const router = express.Router();
const pool = require('../config/db');

// GET /api/fire/dispatch/summary - Operational KPIs and Resource Utilization
router.get('/summary', async (req, res) => {
  try {
    const summaryQuery = `
      SELECT 
        COUNT(*) FILTER (WHERE status NOT IN ('RESOLVED', 'CLOSED')) AS active_incidents,
        COUNT(*) FILTER (WHERE status IN ('REPORTED', 'ACKNOWLEDGED')) AS dispatch_pending,
        COUNT(*) FILTER (WHERE status = 'DISPATCHED') AS dispatched_incidents,
        COUNT(*) FILTER (WHERE status = 'EN_ROUTE') AS en_route,
        COUNT(*) FILTER (WHERE status = 'ON_SCENE') AS on_scene,
        COUNT(*) FILTER (WHERE status = 'CONTAINED') AS contained,
        ROUND(AVG(response_time_minutes)) AS avg_response_time,
        SUM(fire_tenders_required) AS total_tenders_required,
        SUM(fire_tenders_dispatched) AS total_tenders_dispatched,
        SUM(rescue_units_dispatched) AS total_rescue_dispatched,
        SUM(water_tankers_dispatched) AS total_tankers_dispatched
      FROM fire_incidents
      WHERE status NOT IN ('RESOLVED', 'CLOSED') OR created_at >= NOW() - INTERVAL '24 HOURS'
    `;
    const result = await pool.query(summaryQuery);
    res.json(result.rows[0]);
  } catch (error) {
    console.error("[FIRE DISPATCH API] Error fetching dispatch summary:", error);
    res.status(500).json({ error: 'Failed to fetch dispatch summary.' });
  }
});

// GET /api/fire/dispatch/stations - Station Workload
router.get('/stations', async (req, res) => {
  try {
    const stationsQuery = `
      SELECT 
        station_name,
        COUNT(*) FILTER (WHERE status NOT IN ('RESOLVED', 'CLOSED')) AS active_incidents,
        SUM(fire_tenders_dispatched + rescue_units_dispatched + water_tankers_dispatched) AS dispatched_units,
        COUNT(*) FILTER (WHERE status = 'ON_SCENE') AS on_scene_incidents,
        ROUND(AVG(response_time_minutes)) AS avg_response
      FROM fire_incidents
      WHERE status NOT IN ('RESOLVED', 'CLOSED') AND station_name IS NOT NULL
      GROUP BY station_name
      ORDER BY active_incidents DESC
    `;
    const result = await pool.query(stationsQuery);
    res.json(result.rows);
  } catch (error) {
    console.error("[FIRE DISPATCH API] Error fetching station workload:", error);
    res.status(500).json({ error: 'Failed to fetch station workload.' });
  }
});

// GET /api/fire/dispatch - Active Dispatch Queue
router.get('/', async (req, res) => {
  try {
    const { status, severity, incident_type, active_only, search } = req.query;
    let baseQuery = `SELECT * FROM fire_incidents WHERE 1=1`;
    const queryParams = [];
    let paramIndex = 1;

    if (active_only === 'true') {
      baseQuery += ` AND status NOT IN ('RESOLVED', 'CLOSED')`;
    }
    if (status && status !== 'ALL') {
      if (status === 'REPORTED') {
        // Handle "Dispatch Pending" operational group mapping
        baseQuery += ` AND status IN ('REPORTED', 'ACKNOWLEDGED')`;
      } else {
        baseQuery += ` AND status = $${paramIndex++}`;
        queryParams.push(status);
      }
    }
    if (severity && severity !== 'ALL') {
      baseQuery += ` AND severity = $${paramIndex++}`;
      queryParams.push(severity);
    }
    if (incident_type && incident_type !== 'ALL') {
      baseQuery += ` AND incident_type = $${paramIndex++}`;
      queryParams.push(incident_type);
    }
    if (search) {
      baseQuery += ` AND (incident_number ILIKE $${paramIndex} OR location ILIKE $${paramIndex} OR address ILIKE $${paramIndex})`;
      queryParams.push(`%${search}%`);
      paramIndex++;
    }

    // Deterministic prioritization: Severity -> Priority -> Status -> Time
    baseQuery += `
      ORDER BY 
        CASE severity WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MODERATE' THEN 3 WHEN 'LOW' THEN 4 ELSE 5 END,
        CASE priority WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 WHEN 'LOW' THEN 4 ELSE 5 END,
        CASE status 
            WHEN 'REPORTED' THEN 1 
            WHEN 'ACKNOWLEDGED' THEN 2 
            WHEN 'DISPATCHED' THEN 3 
            WHEN 'EN_ROUTE' THEN 4 
            WHEN 'ON_SCENE' THEN 5 
            WHEN 'CONTAINED' THEN 6 
            ELSE 7 
        END,
        reported_at DESC
    `;

    const result = await pool.query(baseQuery, queryParams);
    res.json(result.rows);
  } catch (error) {
    console.error("[FIRE DISPATCH API] Error fetching dispatch queue:", error);
    res.status(500).json({ error: 'Failed to fetch dispatch queue.' });
  }
});

module.exports = router;