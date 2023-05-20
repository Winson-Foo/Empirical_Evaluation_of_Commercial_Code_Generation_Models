// config.js
require('dotenv').config();

module.exports = {
    port: process.env.PORT || 3000,
    hostname: process.env.HOSTNAME || '0.0.0.0'
};

// server.js
const cli = require('next/dist/cli/next-start');
const config = require('./config');

const startServer = () => {
    try {
        cli.nextStart(['-p', config.port, '-H', config.hostname]);
    } catch (error) {
        console.error('Error starting server: ', error);
    }
};

startServer();