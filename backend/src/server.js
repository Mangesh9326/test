require('dotenv').config();
const express = require('express');
const http = require('http');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { initializeSocket } = require('./engine/socketManager');
const { testDatabaseConnection } = require('./config/db');
// Route Imports
const authRoutes = require('./api/routes/authRoutes');
const simulationRoutes = require('./api/routes/simulationRoutes');
const mapRoutes = require('./api/routes/map');
const trafficRoutes = require('./api/routes/trafficRoutes');
const hospitalRoutes = require('./api/routes/hospitalRoutes');
const policeRoutes = require('./api/routes/policeRoutes');
const fireRoutes = require('./api/routes/fireRoutes');
const citizenRoutes = require('./api/routes/citizenRoutes');

const app = express();
const server = http.createServer(app);

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Initialize Socket.IO
initializeSocket(server);

// Static file serving
app.use('/videos', express.static(path.join(__dirname, 'api/public/videos')));
app.use('/uploads', express.static(path.join(__dirname, 'api/public/uploads')));

// Mount API Routes
app.use('/api/auth', authRoutes);
app.use('/api/simulation', simulationRoutes);
app.use('/api/map', mapRoutes);
app.use('/api/traffic', trafficRoutes);
app.use('/api/hospitals', hospitalRoutes);
app.use('/api/police', policeRoutes);
app.use('/api/fire', fireRoutes);
app.use('/api/citizen', citizenRoutes);

// Health Check
app.get('/api/health', (req, res) => {
    res.status(200).json({ status: 'Online', timestamp: new Date().toISOString() });
});

// List videos endpoint
const videoDirectory = path.join(__dirname, 'api/public/videos/list');
app.get('/api/videos/list', (req, res) => {
  fs.readdir(videoDirectory, (err, files) => {
    if (err) {
      console.error('Could not list videos:', err);
      return res.status(500).json({
        error: 'Failed to read videos folder'
      });
    }

    const videoFiles = files.filter(file =>
      file.toLowerCase().endsWith('.mp4')
    );

    res.json(videoFiles);
  });
});

const PORT = process.env.PORT || 5000;

server.listen(PORT, async () => {
    console.log(`[SYSTEM] Server initialized on port ${PORT}`);
    console.log(`[SYSTEM] Environment: ${process.env.NODE_ENV}`);

    try {
        await testDatabaseConnection();
    } catch (error) {
        console.error('[SYSTEM] Database initialization failed');
    }
});