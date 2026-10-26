const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const db = require('../../config/db');

const login = async (req, res) => {
    const { username, password } = req.body;

    if (!username || !password) {
        return res.status(400).json({ message: 'Username and password are required.' });
    }

    try {
        // Fetch user from the database
        const result = await db.query('SELECT * FROM users WHERE username = $1', [username]);
        
        if (result.rows.length === 0) {
            return res.status(404).json({ message: 'User not found.' });
        }

        const user = result.rows[0];

        // Verify password
        const isValidPassword = await bcrypt.compare(password, user.password_hash);
        if (!isValidPassword) {
            return res.status(401).json({ message: 'Invalid credentials.' });
        }

        // Generate JWT Token
        const token = jwt.sign(
            { id: user.id, username: user.username, role: user.role },
            process.env.JWT_SECRET,
            { expiresIn: '8h' } // Standard enterprise session duration
        );

        res.status(200).json({
            message: 'Login successful',
            token,
            user: {
                id: user.id,
                username: user.username,
                role: user.role
            }
        });
    } catch (error) {
        console.error('[ERROR] Login failed:', error);
        res.status(500).json({ message: 'Internal server error during login.' });
    }
};

// Helper function to seed an initial Admin user for testing
const registerInitialAdmin = async (req, res) => {
    try {
        const hashedPassword = await bcrypt.hash('admin123', 10);
        await db.query(
            'INSERT INTO users (username, password_hash, role) VALUES ($1, $2, $3)',
            ['admin', hashedPassword, 'Administrator']
        );
        res.status(201).json({ message: 'Initial Administrator created successfully.' });
    } catch (error) {
        res.status(500).json({ message: 'Failed to create admin.', error: error.message });
    }
};

module.exports = { login, registerInitialAdmin };