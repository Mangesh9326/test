const express = require("express");
const router = express.Router();
const pool = require("../../config/db");
const Groq = require("groq-sdk");
// Note: Requires groq-sdk or standard fetch to Groq API using process.env.GROQ_API_KEY

// Helper for SQL Haversine formula (Distance in KM)
const distanceSQL = (lat, lng, latCol = "latitude", lngCol = "longitude") =>
  `(6371 * acos(cos(radians(${lat})) * cos(radians(${latCol})) * cos(radians(${lngCol}) - radians(${lng})) + sin(radians(${lat})) * sin(radians(${latCol}))))`;

// GET /api/citizen/around-me
router.get("/around-me", async (req, res) => {
  try {
    // Default to Dadar, Mumbai if no location provided
    const lat = parseFloat(req.query.latitude) || 19.0178;
    const lng = parseFloat(req.query.longitude) || 72.8478;
    const radius = parseFloat(req.query.radius) || 5;

    // 1. Fetch Hospitals within radius
    const hospitalsQuery = `
      SELECT hospital_id as id, name, area as location, latitude, longitude, hospital_type as type, emergency_available, 'Hospital' as category, status,
      ${distanceSQL(lat, lng)} AS distance
      FROM hospitals
      WHERE ${distanceSQL(lat, lng)} <= $1
      ORDER BY distance ASC;
    `;

    // 2. Fetch Fire Stations (Corrected to use map_entities)
    const fireQuery = `
      SELECT entity_id as id, name, metadata->>'area' as location, latitude, longitude, 'Fire Station' as type, 'Fire' as category, metadata->>'status' as status,
      ${distanceSQL(lat, lng)} AS distance
      FROM map_entities
      WHERE entity_type = 'fire_station' AND layer_group = 'fire'
      AND ${distanceSQL(lat, lng)} <= $1
      ORDER BY distance ASC;
    `;

    // 3. Fetch Map Entities (Police, Utility, CCTV, Traffic)
    const entitiesQuery = `
      SELECT entity_id as id, name, metadata->>'area' as location, latitude, longitude, entity_type as type, 
      CASE 
        WHEN entity_type ILIKE '%police%' THEN 'Police'
        WHEN entity_type ILIKE '%utility%' THEN 'Utility'
        WHEN entity_type ILIKE '%traffic%' THEN 'Traffic'
        WHEN entity_type ILIKE '%cctv%' THEN 'Traffic'
        ELSE 'Other'
      END as category,
      metadata->>'status' as status,
      ${distanceSQL(lat, lng)} AS distance
      FROM map_entities
      WHERE entity_type IN ('police_station', 'police_unit', 'utility', 'traffic_signal', 'cctv')
      AND ${distanceSQL(lat, lng)} <= $1
      ORDER BY distance ASC;
    `;

    // 4. Fetch Active Incidents (Emergency + Fire)
    // 4. Fetch Active Incidents (General + Fire)
    const incidentsQuery = `
      SELECT 
        incident_id::varchar as id, 
        incident_type as name, 
        COALESCE(timeline_events->>'location', 'Unknown Location') as location, 
        latitude, 
        longitude, 
        incident_type as type, 
        'Incident' as category, 
        severity, 
        status, 
        created_at as time,
        ${distanceSQL(lat, lng)} AS distance
      FROM incidents
      WHERE status NOT ILIKE 'CLOSED' AND status NOT ILIKE 'RESOLVED' 
      AND ${distanceSQL(lat, lng)} <= $1
      
      UNION ALL
      
      SELECT 
        incident_number as id, 
        incident_type as name, 
        location, 
        latitude, 
        longitude, 
        'FIRE_EMERGENCY' as type, 
        'Incident' as category, 
        severity, 
        status, 
        reported_at as time,
        ${distanceSQL(lat, lng)} AS distance
      FROM fire_incidents
      WHERE status NOT IN ('CLOSED', 'RESOLVED') 
      AND ${distanceSQL(lat, lng)} <= $1
      
      ORDER BY distance ASC;
    `;

    const [hospitals, fireStations, entities, incidents] = await Promise.all([
      pool.query(hospitalsQuery, [radius]),
      pool.query(fireQuery, [radius]),
      pool.query(entitiesQuery, [radius]),
      pool.query(incidentsQuery, [radius]),
    ]);

    // Aggregate Services
    const services = [
      ...hospitals.rows,
      ...fireStations.rows,
      ...entities.rows.filter(
        (e) => e.category === "Police" || e.category === "Utility",
      ),
    ].sort((a, b) => a.distance - b.distance);

    const traffic = entities.rows.filter((e) => e.category === "Traffic");
    const activeIncidents = incidents.rows;

    res.json({
      location: { latitude: lat, longitude: lng },
      radius,
      services,
      incidents: activeIncidents,
      traffic,
      alerts: activeIncidents.filter(
        (i) => i.severity === "CRITICAL" || i.severity === "HIGH",
      ),
      isSimulated: true, // Flag for frontend "DEMO DATA" indicator
    });
  } catch (error) {
    console.error("[CITIZEN API] Error fetching around-me data:", error);
    res.status(500).json({ error: "Failed to load nearby city information." });
  }
});

// Citizen Traffic News
router.get("/traffic/news", async (req, res) => {
  try {
    const { lat, lng, radius = 10 } = req.query;

    /*
     * DEMO / MOCK DATA
     *
     * This data is deterministic and intended for development/demo mode.
     * Replace with the actual traffic-news provider/database later.
     *
     * IMPORTANT:
     * Do not present these records as live news.
     */

    const mockNews = [
      {
        news_id: "NEWS-TRAFFIC-001",
        headline: "Truck accident causes traffic disruption near MG Road",
        summary:
          "A reported truck accident has affected traffic movement near MG Road. Motorists may experience congestion around the affected junction.",
        source: "Mumbai Traffic Demo Feed",
        published_time: "10 mins ago",
        published_at: new Date(Date.now() - 10 * 60 * 1000).toISOString(),

        category: "ACCIDENT",
        severity: "HIGH",
        status: "ACTIVE",

        affected_road: "MG Road",
        affected_area: "MG Road Junction",

        latitude: 19.076,
        longitude: 72.8777,

        impact: "Road blockage and heavy congestion",
        estimated_delay_minutes: 20,

        tags: ["truck accident", "road blockage", "heavy traffic"],

        related_incident_id: "INC-TRAFFIC-001",

        is_simulated: true,
      },

      {
        news_id: "NEWS-TRAFFIC-002",
        headline: "Heavy congestion reported on Western Express Highway",
        summary:
          "Traffic is moving slowly on parts of the Western Express Highway due to increased vehicle volume and ongoing infrastructure activity.",
        source: "Mumbai Traffic Demo Feed",
        published_time: "18 mins ago",
        published_at: new Date(Date.now() - 18 * 60 * 1000).toISOString(),

        category: "CONGESTION",
        severity: "MEDIUM",
        status: "ACTIVE",

        affected_road: "Western Express Highway",
        affected_area: "Bandra East",

        latitude: 19.062,
        longitude: 72.849,

        impact: "Slow-moving traffic",
        estimated_delay_minutes: 15,

        tags: ["congestion", "infrastructure work", "traffic delay"],

        related_incident_id: null,

        is_simulated: true,
      },

      {
        news_id: "NEWS-TRAFFIC-003",
        headline: "Traffic diversion reported near Dadar Junction",
        summary:
          "Temporary traffic diversion is affecting vehicle movement near Dadar Junction. Citizens are advised to expect additional travel time.",
        source: "Mumbai Traffic Demo Feed",
        published_time: "25 mins ago",
        published_at: new Date(Date.now() - 25 * 60 * 1000).toISOString(),

        category: "DIVERSION",
        severity: "MEDIUM",
        status: "ACTIVE",

        affected_road: "Dadar Junction",
        affected_area: "Dadar",

        latitude: 19.0178,
        longitude: 72.8478,

        impact: "Traffic diversion and moderate delays",
        estimated_delay_minutes: 12,

        tags: ["traffic diversion", "road traffic"],

        related_incident_id: "INC-TRAFFIC-002",

        is_simulated: true,
      },

      {
        news_id: "NEWS-TRAFFIC-004",
        headline: "Traffic movement affected near Sion Circle",
        summary:
          "Vehicle movement is slower than normal near Sion Circle. Congestion is currently affecting surrounding roads.",
        source: "Mumbai Traffic Demo Feed",
        published_time: "32 mins ago",
        published_at: new Date(Date.now() - 32 * 60 * 1000).toISOString(),

        category: "CONGESTION",
        severity: "LOW",
        status: "MONITORING",

        affected_road: "Sion Circle",
        affected_area: "Sion",

        latitude: 19.0437,
        longitude: 72.861,

        impact: "Moderate traffic buildup",
        estimated_delay_minutes: 8,

        tags: ["traffic buildup", "slow traffic"],

        related_incident_id: null,

        is_simulated: true,
      },

      {
        news_id: "NEWS-TRAFFIC-005",
        headline: "Road maintenance may cause delays near Andheri",
        summary:
          "Road maintenance activity is affecting traffic movement in parts of Andheri. Delays may occur during the maintenance period.",
        source: "Mumbai Traffic Demo Feed",
        published_time: "45 mins ago",
        published_at: new Date(Date.now() - 45 * 60 * 1000).toISOString(),

        category: "ROAD_WORK",
        severity: "LOW",
        status: "ACTIVE",

        affected_road: "Andheri Link Road",
        affected_area: "Andheri",

        latitude: 19.1197,
        longitude: 72.8468,

        impact: "Reduced road capacity",
        estimated_delay_minutes: 10,

        tags: ["road work", "maintenance", "traffic delay"],

        related_incident_id: null,

        is_simulated: true,
      },
    ];

    /*
     * Optional location filtering.
     *
     * For now this returns all mock records.
     * When the real news API is integrated, use lat/lng/radius
     * to retrieve relevant traffic news.
     */

    res.json({
      success: true,

      location: {
        latitude: lat ? Number(lat) : null,
        longitude: lng ? Number(lng) : null,
        radius_km: Number(radius),
      },

      source: "MOCK_TRAFFIC_NEWS",

      is_simulated: true,

      news: mockNews,

      count: mockNews.length,
    });
  } catch (error) {
    console.error("News API Error:", error);

    res.status(500).json({
      success: false,
      error: "Failed to fetch traffic news.",
    });
  }
});

// GET /api/citizen/alerts
router.get("/alerts", async (req, res) => {
  try {
    const userLat = parseFloat(req.query.latitude) || 19.0178;
    const userLng = parseFloat(req.query.longitude) || 72.8478;

    const query = `
      SELECT 
        incident_id::varchar as alert_id, 
        CASE 
          WHEN incident_type ILIKE '%fire%' THEN 'FIRE'
          WHEN incident_type ILIKE '%traffic%' OR incident_type ILIKE '%accident%' OR incident_type ILIKE '%collision%' THEN 'TRAFFIC'
          WHEN incident_type ILIKE '%flood%' OR incident_type ILIKE '%rain%' OR incident_type ILIKE '%water%' THEN 'WEATHER'
          WHEN incident_type ILIKE '%police%' OR incident_type ILIKE '%robbery%' THEN 'POLICE'
          ELSE 'GENERAL'
        END as category,
        UPPER(severity) as severity, 
        incident_type as title, 
        
        -- FIXED: Removed "description" and safely parse the JSONB timeline_events instead
        COALESCE(timeline_events->0->>'description', incident_type || ' reported in the area') as message, 
        COALESCE(timeline_events->>'location', 'Unknown Location') as location, 
        
        latitude, 
        longitude, 
        status, 
        created_at,
        'Smart City Operations' as source,
        true as is_simulated,
        ROUND(${distanceSQL(userLat, userLng)}::numeric, 1) AS distance
      FROM incidents
      WHERE status NOT ILIKE 'CLOSED' AND status NOT ILIKE 'RESOLVED'
      ORDER BY 
        CASE UPPER(severity) WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MODERATE' THEN 3 WHEN 'MEDIUM' THEN 3 WHEN 'LOW' THEN 4 ELSE 5 END,
        created_at DESC;
    `;
    const result = await pool.query(query);

    // Deterministic Mock Fallback for DEMO mode if database is entirely empty
    let alerts = result.rows;
    if (alerts.length === 0) {
      alerts = [
        {
          alert_id: "DEMO-001",
          category: "TRAFFIC",
          severity: "CRITICAL",
          title: "MG Road blocked",
          message:
            "Truck accident has affected traffic near MG Road. Road blockage causing significant delay.",
          location: "MG Road Junction",
          latitude: 19.076,
          longitude: 72.8777,
          status: "ACTIVE",
          created_at: new Date(Date.now() - 1000 * 60 * 10).toISOString(),
          source: "Traffic Incident",
          is_simulated: true,
          distance: 1.4,
        },
        {
          alert_id: "DEMO-002",
          category: "WEATHER",
          severity: "HIGH",
          title: "Heavy rainfall warning",
          message:
            "Heavy rainfall may cause waterlogging in low-lying areas. Exercise caution.",
          location: "Mumbai South",
          latitude: 18.922,
          longitude: 72.8347,
          status: "ACTIVE",
          created_at: new Date(Date.now() - 1000 * 60 * 45).toISOString(),
          source: "Weather Dept",
          is_simulated: true,
          distance: 4.2,
        },
      ];
    }

    res.json({ alerts, isSimulated: true });
  } catch (error) {
    console.error("[CITIZEN ALERTS API] Error fetching alerts:", error);
    res.status(500).json({ error: "Failed to load alerts." });
  }
});

// GET /api/citizen/services
router.get("/services", async (req, res) => {
  try {
    const lat = parseFloat(req.query.latitude) || 19.0178;
    const lng = parseFloat(req.query.longitude) || 72.8478;
    const radius = parseFloat(req.query.radius) || 10;
    const search = req.query.search ? req.query.search.toLowerCase() : null;
    const categoryFilter = req.query.category || null;

    // 1. Fetch Hospitals
    const hospitalsQuery = `
      SELECT 
        hospital_id as service_id, 
        name, 
        'Healthcare' as category, 
        'Hospital' as department,
        hospital_type || ' providing medical services.' as description,
        area, 
        area as address, 
        latitude, 
        longitude, 
        status, 
        emergency_available,
        true as is_simulated,
        ROUND(${distanceSQL(lat, lng)}::numeric, 2) AS distance_km
      FROM hospitals
      WHERE ${distanceSQL(lat, lng)} <= $1
    `;

    // 2. Fetch Map Entities (Police, Fire, Utility, Transport)
    const entitiesQuery = `
      SELECT 
        entity_id as service_id, 
        name, 
        CASE 
          WHEN entity_type ILIKE '%police%' THEN 'Police'
          WHEN entity_type ILIKE '%fire%' THEN 'Fire'
          WHEN entity_type ILIKE '%utility%' THEN 'Utilities'
          WHEN entity_type ILIKE '%traffic%' THEN 'Traffic'
          WHEN entity_type ILIKE '%transport%' THEN 'Transport'
          ELSE 'Civic'
        END as category,
        entity_type as department,
        COALESCE(metadata->>'description', 'Municipal service facility.') as description,
        COALESCE(metadata->>'area', 'Unknown Area') as area,
        COALESCE(metadata->>'address', metadata->>'area') as address,
        latitude, 
        longitude, 
        COALESCE(metadata->>'status', 'Unknown') as status,
        CASE WHEN entity_type ILIKE ANY(ARRAY['%police%', '%fire%']) THEN true ELSE false END as emergency_available,
        true as is_simulated,
        ROUND(${distanceSQL(lat, lng)}::numeric, 2) AS distance_km
      FROM map_entities
      WHERE entity_type IN ('police_station', 'police_unit', 'fire_station', 'utility', 'transport_hub', 'civic_center')
      AND ${distanceSQL(lat, lng)} <= $1
    `;

    const [hospitalsRes, entitiesRes] = await Promise.all([
      pool.query(hospitalsQuery, [radius]),
      pool.query(entitiesQuery, [radius]),
    ]);

    let services = [...hospitalsRes.rows, ...entitiesRes.rows];

    // Apply Search Filter
    if (search) {
      services = services.filter(
        (s) =>
          s.name.toLowerCase().includes(search) ||
          s.category.toLowerCase().includes(search) ||
          (s.area && s.area.toLowerCase().includes(search)),
      );
    }

    // Apply Category Filter
    if (categoryFilter && categoryFilter !== "All") {
      services = services.filter(
        (s) => s.category.toLowerCase() === categoryFilter.toLowerCase(),
      );
    }

    // Sort by distance
    services.sort((a, b) => a.distance_km - b.distance_km);

    res.json({
      success: true,
      services,
      location: { latitude: lat, longitude: lng },
      radius,
      isSimulated: true,
    });
  } catch (error) {
    console.error("[CITIZEN SERVICES API] Error fetching services:", error);
    res
      .status(500)
      .json({ success: false, error: "Failed to load city services." });
  }
});
// POST /api/citizen/assistant - Secure Groq AI Integration
const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

router.post("/assistant", async (req, res) => {
  try {
    const { message, context } = req.body;

    if (!message || typeof message !== "string") {
      return res.status(400).json({
        error: "Message is required",
      });
    }

    const systemPrompt = `
You are the Smart City Citizen Assistant.

Respond concisely and clearly based strictly on the
provided platform context.

Do not invent incidents, police actions, traffic
statuses, or other information.

If the information is not available in the context,
say:
"Based on available platform data, this information
is not currently available."

PLATFORM CONTEXT:
${JSON.stringify(context)}
`;

    const completion = await groq.chat.completions.create({
      model: process.env.GROQ_MODEL,
      temperature: 0.1,
      messages: [
        {
          role: "system",
          content: systemPrompt,
        },
        {
          role: "user",
          content: message,
        },
      ],
    });

    const reply = completion.choices?.[0]?.message?.content;

    if (!reply) {
      return res.status(502).json({
        error: "No response received from AI service",
      });
    }

    return res.json({
      reply,
    });
  } catch (error) {
    console.error("[CITIZEN AI]", error);

    return res.status(500).json({
      error: "AI assistant is temporarily unavailable.",
    });
  }
});
module.exports = router;
