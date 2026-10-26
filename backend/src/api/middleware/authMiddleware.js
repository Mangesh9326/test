const jwt = require('jsonwebtoken');

// Verifies the JWT token
const verifyToken = (req, res, next) => {
    const token = req.headers['authorization'];

    if (!token) {
        return res.status(403).json({ message: 'No token provided. Access denied.' });
    }

    try {
        // Bearer token format parsing
        const actualToken = token.split(' ')[1];
        const decoded = jwt.verify(actualToken, process.env.JWT_SECRET);
        req.user = decoded; // Attach user payload to the request
        next();
    } catch (err) {
        return res.status(401).json({ message: 'Unauthorized. Invalid token.' });
    }
};

// Verifies if the user holds one of the allowed roles
const verifyRole = (allowedRoles) => {
    return (req, res, next) => {
        if (!req.user || !allowedRoles.includes(req.user.role)) {
            return res.status(403).json({ message: 'Forbidden. Insufficient permissions.' });
        }
        next();
    };
};

module.exports = { verifyToken, verifyRole };