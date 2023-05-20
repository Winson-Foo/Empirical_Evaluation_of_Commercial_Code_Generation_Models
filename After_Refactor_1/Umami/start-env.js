require('dotenv').config();
const cli = require('next/dist/cli/next-start');

function startServer(port, hostname) {
  cli.nextStart(['-p', port, '-H', hostname]);
}

const PORT = process.env.PORT || 3000;
const HOSTNAME = process.env.HOSTNAME || '0.0.0.0';

startServer(PORT, HOSTNAME);