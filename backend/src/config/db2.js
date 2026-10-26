const { Pool } = require("pg");
require("dotenv").config();

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,

  ssl: {
    rejectUnauthorized: false,
  },
});

pool.on("error", (err) => {
  console.error("[DB ERROR] Unexpected PostgreSQL pool error:", err);
});

async function testDatabaseConnection() {
  try {
    const client = await pool.connect();

    console.log("[SYSTEM] Neon PostgreSQL Database Connected");

    client.release();
  } catch (error) {
    console.error("[DB ERROR] PostgreSQL connection failed:", error.message);

    throw error;
  }
}

module.exports = {
  query: (text, params) => pool.query(text, params),
  pool,
  testDatabaseConnection,
};