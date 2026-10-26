const express = require('express');
const router = express.Router();
const authController = require('../controllers/authController');

// Actual Login Route
router.post('/login', authController.login);

// Development Only: Route to create the first admin user
router.post('/setup-admin', authController.registerInitialAdmin);

module.exports = router;