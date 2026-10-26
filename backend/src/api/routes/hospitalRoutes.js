const express = require("express");
const router = express.Router();
const pool = require("../../config/db");
const HospitalAgent = require("../../agents/HospitalAgent");

// UUID Validation Middleware
const isValidUUID = (uuid) =>
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(uuid);

// ==========================================
// HOSPITAL CORE API ENDPOINTS
// ==========================================

// GET /api/network-summary
// Calculates live KPIs directly from the database without hardcoding
router.get("/network-summary", async (req, res) => {
  try {
    const query = `
      SELECT 
        COUNT(h.hospital_id)::int AS total_hospitals,
        COUNT(h.hospital_id) FILTER (WHERE h.status = 'Active')::int AS operational_hospitals,
        COUNT(h.hospital_id) FILTER (WHERE h.emergency_available = true)::int AS emergency_ready_hospitals,
        COUNT(m.hospital_id) FILTER (WHERE m.capacity_level = 'HIGH')::int AS high_pressure_hospitals,
        COUNT(m.hospital_id) FILTER (WHERE m.capacity_level = 'CRITICAL')::int AS critical_hospitals,
        COALESCE(SUM(m.total_beds), 0)::int AS total_beds,
        COALESCE(SUM(m.available_beds), 0)::int AS available_beds,
        COALESCE(SUM(m.icu_total), 0)::int AS total_icu,
        COALESCE(SUM(m.icu_available), 0)::int AS available_icu
      FROM hospitals h
      LEFT JOIN hospital_monitoring m ON h.hospital_id = m.hospital_id
    `;
    const result = await pool.query(query);
    res.json(result.rows[0]);
  } catch (error) {
    console.error("[HOSPITAL API] Error fetching network summary:", error);
    res.status(500).json({ error: "Failed to fetch network summary" });
  }
});

// GET /api
// Lists hospitals with comprehensive data, pagination, and parameterized filters
router.get("", async (req, res) => {
  try {
    const { page = 1, limit = 50, search, area, status } = req.query;
    const offset = (Math.max(1, page) - 1) * Math.max(1, limit);

    let conditions = [];
    let values = [];
    let paramIndex = 1;

    if (area && area !== "All") {
      conditions.push(`h.area = $${paramIndex++}`);
      values.push(area);
    }
    if (status && status !== "All") {
      conditions.push(`h.status = $${paramIndex++}`);
      values.push(status);
    }
    if (search) {
      conditions.push(
        `(h.name ILIKE $${paramIndex} OR h.area ILIKE $${paramIndex})`,
      );
      values.push(`%${search}%`);
      paramIndex++;
    }

    const whereClause =
      conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";

    const countQuery = `SELECT COUNT(*) FROM hospitals h ${whereClause}`;
    const totalResult = await pool.query(countQuery, values);
    const total = parseInt(totalResult.rows[0].count);

    const dataQuery = `
      SELECT 
        h.hospital_id, h.name, h.area, h.latitude, h.longitude, h.hospital_type, h.emergency_available, h.status,
        m.total_beds, m.occupied_beds, m.available_beds,
        m.icu_total, m.icu_occupied, m.icu_available,
        m.emergency_capacity, m.emergency_occupancy, m.emergency_level,
        m.ambulance_queue, m.average_wait_minutes,
        m.patient_inflow, m.predicted_inflow,
        m.last_observed_at,
        
        -- Explicit Data-Unavailable States for Missing Columns
        NULL AS trauma_capacity,
        NULL AS operating_rooms,
        NULL AS ventilators_total,
        NULL AS ventilators_available,
        NULL AS oxygen_status,
        NULL AS blood_supply_status,
        NULL AS diagnostic_capacity,
        NULL AS emergency_doctors_available,
        NULL AS ambulances_nearby

      FROM hospitals h
      LEFT JOIN hospital_monitoring m ON h.hospital_id = m.hospital_id
      ${whereClause}
      ORDER BY 
        CASE m.capacity_level WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MODERATE' THEN 3 ELSE 4 END,
        h.name ASC
      LIMIT $${paramIndex++} OFFSET $${paramIndex++}
    `;

    const result = await pool.query(dataQuery, [...values, limit, offset]);

    res.json({
      data: result.rows,
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total,
        totalPages: Math.ceil(total / limit),
      },
    });
  } catch (error) {
    console.error("[HOSPITAL API] Error fetching hospitals:", error);
    res.status(500).json({ error: "Failed to fetch hospitals" });
  }
});

// GET /api/monitoring
router.get("/monitoring", async (req, res) => {
  try {
    const query = `
      SELECT h.name, h.area, m.* 
      FROM hospital_monitoring m
      JOIN hospitals h ON m.hospital_id = h.hospital_id
      ORDER BY 
        CASE m.capacity_level WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MODERATE' THEN 3 ELSE 4 END, 
        h.name ASC
    `;
    const result = await pool.query(query);
    res.json(result.rows);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch monitoring data" });
  }
});

// GET /api/beds
router.get("/beds", async (req, res) => {
  try {
    const query = `
      SELECT h.hospital_id, h.name, h.area, m.total_beds, m.occupied_beds, m.available_beds, m.icu_total, m.icu_occupied, m.icu_available, m.last_observed_at
      FROM hospitals h JOIN hospital_monitoring m ON h.hospital_id = m.hospital_id
    `;
    const result = await pool.query(query);
    res.json(result.rows);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch bed data" });
  }
});

// GET /api/emergency
router.get("/emergency", async (req, res) => {
  try {
    const query = `
      SELECT h.hospital_id, h.name, h.status AS hospital_status,
             m.emergency_capacity, m.emergency_occupancy, m.ambulance_queue,
             m.average_wait_minutes, m.emergency_level, m.capacity_level,
             m.patient_inflow, m.predicted_inflow, m.last_observed_at
      FROM hospitals h JOIN hospital_monitoring m ON h.hospital_id = m.hospital_id
      WHERE h.emergency_available = true
    `;
    const result = await pool.query(query);
    res.json(result.rows);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch emergency data" });
  }
});

// GET /api/capacity
router.get("/capacity", async (req, res) => {
  try {
    const query = `SELECT h.name, h.area, m.capacity_level, m.available_beds, m.icu_available, m.last_observed_at FROM hospitals h JOIN hospital_monitoring m ON h.hospital_id = m.hospital_id`;
    const result = await pool.query(query);
    res.json(result.rows);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch capacity data" });
  }
});

// GET /api/patient-flow
router.get("/patient-flow", async (req, res) => {
  try {
    const query = `SELECT h.name, m.patient_inflow, m.predicted_inflow, m.ambulance_queue, m.average_wait_minutes, m.last_observed_at FROM hospitals h JOIN hospital_monitoring m ON h.hospital_id = m.hospital_id`;
    const result = await pool.query(query);
    res.json(result.rows);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch patient flow data" });
  }
});

// GET /api/resources
router.get("/resources", async (req, res) => {
  try {
    const query = `
      SELECT 
        h.name AS hospital_name, 
        h.area, 
        r.resource_type, 
        r.total_capacity, 
        r.in_use_qty, 
        r.available_qty, 
        r.threshold_percent, 
        r.status, 
        r.last_updated 
      FROM hospital_resources r
      JOIN hospitals h ON r.hospital_id = h.hospital_id
      ORDER BY 
        CASE r.status WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MONITOR' THEN 3 ELSE 4 END,
        h.name ASC
    `;
    const result = await pool.query(query);
    res.json(result.rows);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch resource data" });
  }
});

// GET /api/ambulances
router.get("/ambulances", async (req, res) => {
  try {
    const query = `
      SELECT 
        a.ambulance_id, a.incident_id, a.current_latitude, a.current_longitude,
        a.priority, a.eta_minutes, a.status, a.last_updated,
        h.name AS recommended_hospital, h.hospital_id,
        m.capacity_level AS hospital_capacity_status,
        m.emergency_level AS hospital_emergency_status
      FROM ambulance_operations a
      LEFT JOIN hospitals h ON a.destination_hospital_id = h.hospital_id
      LEFT JOIN hospital_monitoring m ON h.hospital_id = m.hospital_id
      ORDER BY CASE a.priority WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 ELSE 3 END, a.eta_minutes ASC
    `;
    const result = await pool.query(query);
    res.json(result.rows);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch ambulance operations" });
  }
});

// GET /api/hospitals/incidents
router.get("/incidents", async (req, res) => {
  try {
    const query = `
      SELECT
        i.incident_id,
        i.incident_type,

        -- Human-readable location from incident JSON
        COALESCE(
          i.timeline_events->>'location',
          CONCAT(
            ROUND(i.latitude::numeric, 5),
            ', ',
            ROUND(i.longitude::numeric, 5)
          )
        ) AS location,

        i.latitude,
        i.longitude,

        i.created_at AS reported_at,

        -- Incident severity
        i.severity,

        -- Frontend compatibility
        i.severity AS priority,

        i.status AS incident_status,

        -- ==========================================
        -- PATIENT INFORMATION
        -- ==========================================

        COALESCE(
          NULLIF(i.timeline_events->>'patient_count', '')::integer,
          0
        ) AS patient_count,

        COALESCE(
          NULLIF(i.timeline_events->>'critical_patient_count', '')::integer,
          0
        ) AS critical_patient_count,

        COALESCE(
          NULLIF(i.timeline_events->>'serious_patient_count', '')::integer,
          0
        ) AS serious_patient_count,

        COALESCE(
          NULLIF(i.timeline_events->>'moderate_patient_count', '')::integer,
          0
        ) AS moderate_patient_count,

        -- ==========================================
        -- AMBULANCE REQUIREMENT
        -- ==========================================

        COALESCE(
          NULLIF(i.timeline_events->>'ambulances_required', '')::integer,
          0
        ) AS ambulances_required,

        COUNT(DISTINCT a.id)::integer AS ambulances_assigned,

        -- ==========================================
        -- RECEIVING HOSPITAL
        -- ==========================================

        MAX(h.name) AS receiving_hospital_name,

        (
  ARRAY_AGG(h.hospital_id ORDER BY a.id)
  FILTER (WHERE h.hospital_id IS NOT NULL)
)[1] AS receiving_hospital_id,

        MAX(h.area) AS receiving_hospital_area,

        -- ==========================================
        -- HOSPITAL OPERATIONAL STATUS
        -- ==========================================

        MAX(h.status) AS hospital_status,

        (
  ARRAY_AGG(h.emergency_available ORDER BY a.id)
  FILTER (WHERE h.emergency_available IS NOT NULL)
)[1] AS hospital_emergency_available,

        MAX(m.emergency_level) AS hospital_emergency_status,

        MAX(m.capacity_level) AS hospital_capacity_status,

        MAX(m.available_beds)::integer AS available_beds,

        MAX(m.icu_available)::integer AS icu_available,

        MAX(m.emergency_capacity)::integer AS emergency_capacity,

        MAX(m.emergency_occupancy)::integer AS emergency_occupancy,

        MAX(m.ambulance_queue)::integer AS hospital_ambulance_queue,

        MAX(m.average_wait_minutes)::integer AS hospital_average_wait_minutes,

        -- ==========================================
        -- RESPONSE INFORMATION
        -- ==========================================

        COALESCE(
          i.timeline_events->>'response_status',
          CASE
            WHEN COUNT(DISTINCT a.id) > 0 THEN 'AMBULANCE_ASSIGNED'
            ELSE 'AWAITING_ASSIGNMENT'
          END
        ) AS response_status,

        COALESCE(
          NULLIF(
            i.timeline_events->>'estimated_scene_arrival_minutes',
            ''
          )::integer,
          NULL
        ) AS estimated_scene_arrival_minutes,

        COALESCE(
          NULLIF(
            i.timeline_events->>'estimated_hospital_eta_minutes',
            ''
          )::integer,
          MAX(a.eta_minutes)
        ) AS estimated_hospital_eta_minutes,

        -- ==========================================
        -- CALCULATED ETA TIMESTAMPS
        -- ==========================================

        CASE
          WHEN NULLIF(
            i.timeline_events->>'estimated_scene_arrival_minutes',
            ''
          ) IS NOT NULL
          THEN
            i.created_at +
            (
              NULLIF(
                i.timeline_events->>'estimated_scene_arrival_minutes',
                ''
              )::integer * INTERVAL '1 minute'
            )
          ELSE NULL
        END AS estimated_scene_arrival,

        CASE
          WHEN COALESCE(
            NULLIF(
              i.timeline_events->>'estimated_hospital_eta_minutes',
              ''
            )::integer,
            MAX(a.eta_minutes)
          ) IS NOT NULL
          THEN
            CURRENT_TIMESTAMP +
            (
              COALESCE(
                NULLIF(
                  i.timeline_events->>'estimated_hospital_eta_minutes',
                  ''
                )::integer,
                MAX(a.eta_minutes)
              ) * INTERVAL '1 minute'
            )
          ELSE NULL
        END AS hospital_eta,

        -- ==========================================
        -- INCIDENT DETAILS
        -- ==========================================

        i.timeline_events->>'description' AS incident_description,

        i.timeline_events->>'scenario_id' AS scenario_id,

        COALESCE(
          (i.timeline_events->>'confidence')::numeric,
          0
        ) AS detection_confidence,

        i.assigned_departments,

        i.timeline_events->'timeline' AS operational_timeline,

        -- ==========================================
        -- AMBULANCE SUMMARY
        -- ==========================================

        COALESCE(
          json_agg(
            DISTINCT jsonb_build_object(
              'ambulance_id', a.ambulance_id,
              'priority', a.priority,
              'status', a.status,
              'eta_minutes', a.eta_minutes,
              'current_latitude', a.current_latitude,
              'current_longitude', a.current_longitude,
              'hospital_name', h.name
            )
          ) FILTER (WHERE a.id IS NOT NULL),
          '[]'::json
        ) AS ambulances

      FROM incidents i

      -- ==========================================
      -- AMBULANCE ASSIGNMENTS
      -- ==========================================

      LEFT JOIN ambulance_operations a
        ON a.incident_id = i.incident_id::text

      -- ==========================================
      -- RECEIVING HOSPITAL
      -- ==========================================

      LEFT JOIN hospitals h
        ON a.destination_hospital_id = h.hospital_id

      -- ==========================================
      -- HOSPITAL MONITORING
      -- ==========================================

      LEFT JOIN hospital_monitoring m
        ON h.hospital_id = m.hospital_id

      GROUP BY
        i.incident_id,
        i.incident_type,
        i.latitude,
        i.longitude,
        i.created_at,
        i.severity,
        i.status,
        i.timeline_events,
        i.assigned_departments

      ORDER BY
        CASE UPPER(i.severity)
          WHEN 'CRITICAL' THEN 1
          WHEN 'HIGH' THEN 2
          WHEN 'MODERATE' THEN 3
          WHEN 'LOW' THEN 4
          ELSE 5
        END,
        i.created_at DESC
    `;

    const result = await pool.query(query);

    res.json(result.rows);
  } catch (error) {
    console.error("[HOSPITAL API] Error fetching emergency incidents:", error);

    res.status(500).json({
      error: "Failed to fetch emergency incidents",
      details:
        process.env.NODE_ENV === "development" ? error.message : undefined,
    });
  }
});
// ==========================================
// AGENT API ENDPOINTS
// ==========================================

router.get("/hospital/agent/overview", async (req, res) => {
  try {
    const assessment = await HospitalAgent.getNetworkAssessment();
    res.json(assessment);
  } catch (error) {
    res.status(500).json({ error: "Failed to generate network assessment." });
  }
});

router.get("/hospital/agent/decisions", async (req, res) => {
  try {
    const { page = 1, limit = 20, status = "ACTIVE" } = req.query;
    const offset = (Math.max(1, page) - 1) * Math.max(1, limit);

    const query = `
      SELECT d.*, h.name as hospital_name, h.area 
      FROM hospital_agent_decisions d
      JOIN hospitals h ON d.hospital_id = h.hospital_id
      WHERE d.status = $1
      ORDER BY d.created_at DESC
      LIMIT $2 OFFSET $3
    `;
    const result = await pool.query(query, [status, limit, offset]);
    res.json({ data: result.rows, page: Number(page) });
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch decisions" });
  }
});

router.get("/hospital/agent/active", async (req, res) => {
  try {
    const query = `
      SELECT d.*, h.name as hospital_name 
      FROM hospital_agent_decisions d
      JOIN hospitals h ON d.hospital_id = h.hospital_id
      WHERE d.status = 'ACTIVE'
      ORDER BY d.confidence DESC
    `;
    const result = await pool.query(query);
    res.json(result.rows);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch active decisions" });
  }
});

router.get("/hospital/agent/hospital/:id", async (req, res) => {
  try {
    if (!isValidUUID(req.params.id))
      return res.status(400).json({ error: "Invalid hospital ID format" });

    const query = `
      SELECT h.*, m.* 
      FROM hospitals h
      LEFT JOIN hospital_monitoring m ON h.hospital_id = m.hospital_id
      WHERE h.hospital_id = $1
    `;
    const result = await pool.query(query, [req.params.id]);
    if (result.rows.length === 0)
      return res.status(404).json({ error: "Hospital not found" });

    const assessment = HospitalAgent.evaluateHospital(result.rows[0]);
    res.json(assessment);
  } catch (error) {
    res.status(500).json({ error: "Failed to assess hospital" });
  }
});

// ==========================================
// WILDCARD ROUTES (MUST BE LAST)
// ==========================================

// GET /api/:id
router.get("/:id", async (req, res) => {
  try {
    if (!isValidUUID(req.params.id))
      return res.status(400).json({ error: "Invalid hospital ID format" });

    const query = `
      SELECT 
        h.hospital_id, h.name, h.area, h.latitude, h.longitude, h.hospital_type, h.emergency_available, h.status,
        m.total_beds, m.occupied_beds, m.available_beds,
        m.icu_total, m.icu_occupied, m.icu_available,
        m.emergency_capacity, m.emergency_occupancy, m.emergency_level,
        m.ambulance_queue, m.average_wait_minutes,
        m.patient_inflow, m.predicted_inflow,
        m.capacity_level, m.last_observed_at,
        
        -- Explicit Data-Unavailable States
        NULL AS trauma_capacity, NULL AS operating_rooms, NULL AS ventilators_total,
        NULL AS ventilators_available, NULL AS oxygen_status, NULL AS blood_supply_status,
        NULL AS diagnostic_capacity, NULL AS emergency_doctors_available, NULL AS ambulances_nearby

      FROM hospitals h
      LEFT JOIN hospital_monitoring m ON h.hospital_id = m.hospital_id
      WHERE h.hospital_id = $1
    `;
    const result = await pool.query(query, [req.params.id]);

    if (result.rows.length === 0)
      return res.status(404).json({ error: "Hospital not found" });
    res.json(result.rows[0]);
  } catch (error) {
    console.error("[HOSPITAL API] Error fetching hospital by ID:", error);
    res.status(500).json({ error: "Failed to fetch hospital details" });
  }
});

// POST /api/hospitals/agent/evaluate-emergency
// Phase 7: AI Agent Coordination Simulation Endpoint
router.post("/agent/evaluate-emergency", async (req, res) => {
  try {
    const { incident_type, location, severity, destination_hospital } =
      req.body;

    // 1. Hospital Agent Deterministic Evaluation
    let hospitalDecision = null;
    if (destination_hospital) {
      const hQuery = `
        SELECT h.*, m.* 
        FROM hospitals h
        LEFT JOIN hospital_monitoring m ON h.hospital_id = m.hospital_id
        WHERE h.name = $1 LIMIT 1
      `;
      const hRes = await pool.query(hQuery, [destination_hospital]);
      if (hRes.rows.length > 0) {
        const hData = hRes.rows[0];
        // Execute existing HospitalAgent evaluation logic
        hospitalDecision = HospitalAgent.evaluateHospital(hData);
      }
    }

    // Fallback if exact hospital mapping fails but we still need deterministic simulation output
    if (!hospitalDecision) {
      hospitalDecision = {
        hospital_name: destination_hospital || "Nearest Facility",
        current_condition: "HIGH",
        decision_type: "CAPACITY_PREPARATION",
        decision: "Prepare alternate receiving capacity.",
        recommendation: `Review alternate receiving capacity. Maintain standard routing to ${destination_hospital || "nearest facility"}.`,
        expected_impact: "Reduce projected emergency overload.",
        reason: "High patient inflow projected based on incident severity.",
        confidence: 87,
      };
    }

    // 2. Traffic Agent Deterministic Simulation Output
    const trafficDecision = {
      recommended_route: `Primary Arterial to ${destination_hospital || "Destination"}`,
      estimated_travel_time: severity === "CRITICAL" ? "9 min" : "14 min",
      traffic_condition: severity === "CRITICAL" ? "MODERATE" : "HIGH",
      route_risk: severity === "CRITICAL" ? "Elevated" : "Standard",
      corridor_recommendation:
        severity === "CRITICAL"
          ? "Simulated Green Corridor Active"
          : "Standard Routing Recommended",
      decision:
        severity === "CRITICAL"
          ? "Prioritize emergency corridor"
          : "Monitor traffic flow",
      confidence: severity === "CRITICAL" ? 94 : 88,
    };

    res.json({
      hospital: hospitalDecision,
      traffic: trafficDecision,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("[HOSPITAL API] Error evaluating emergency:", error);
    res.status(500).json({ error: "Failed to evaluate emergency" });
  }
});
module.exports = router;
