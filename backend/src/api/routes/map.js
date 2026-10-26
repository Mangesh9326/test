const express = require('express');
const router = express.Router();
const db = require("../../config/db");

// GET /api/map/entities
router.get('/entities', async (req, res) => {
    try {
        const staticEntities = await db.query('SELECT * FROM map_entities');
        const vehicles = await db.query('SELECT * FROM vehicles');
        const incidents = await db.query("SELECT * FROM incidents WHERE status = 'active'");

        res.json({
            staticEntities: staticEntities.rows,
            vehicles: vehicles.rows,
            incidents: incidents.rows
        });
    } catch (err) {
        console.error('[API] Error fetching map entities:', err);
        res.status(500).json({ error: 'Failed to fetch map data' });
    }
});

router.get('/data', async (req, res) => {
    const staticEntities = await db.query('SELECT * FROM map_entities');
    const vehicles = await db.query('SELECT * FROM vehicles');
    const incidents = await db.query('SELECT * FROM incidents WHERE status != \'resolved\'');
    res.json({ static: staticEntities.rows, vehicles: vehicles.rows, incidents: incidents.rows });
});

module.exports = router;