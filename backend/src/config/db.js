const { Pool } = require('pg');
require('dotenv').config();

const pool = new Pool({
    user: process.env.DB_USER,
    host: process.env.DB_HOST,
    database: process.env.DB_NAME,
    password: process.env.DB_PASSWORD,
    port: process.env.DB_PORT,
});

pool.on('error', (err) => {
    console.error('[DB ERROR] Unexpected PostgreSQL pool error:', err);
});

async function testDatabaseConnection() {
    try {
        const client = await pool.connect();

        console.log('[SYSTEM] PostgreSQL Database Connected');

        client.release();
    } catch (error) {
        console.error('[DB ERROR] PostgreSQL connection failed:', error.message);
        throw error;
    }
}

module.exports = {
    query: (text, params) => pool.query(text, params),
    pool,
    testDatabaseConnection,
};