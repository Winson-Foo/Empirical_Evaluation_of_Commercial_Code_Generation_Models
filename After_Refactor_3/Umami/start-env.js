require('dotenv').config();
const cli = require('next/dist/cli/next-start');

const {PORT = 3000, HOSTNAME = '0.0.0.0'} = process.env;
const nextStartOptions = {
  port: PORT,
  hostname: HOSTNAME,
};

cli.nextStart(nextStartOptions);