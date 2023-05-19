/* eslint-disable no-console */
require('dotenv').config();
const fs = require('fs');
const path = require('path');

const updateTrackerEndpoint = (endpoint) => {
  const file = path.resolve(__dirname, '../public/script.js');
  const tracker = fs.readFileSync(file);

  fs.writeFileSync(
    path.resolve(file),
    tracker.toString().replace(/"\/api\/send"/g, `"${endpoint}"`),
  );
};

const main = () => {
  const endPoint = process.env.COLLECT_API_ENDPOINT;
  if (endPoint) {
    updateTrackerEndpoint(endPoint);
    console.log(`Updated tracker endpoint: ${endPoint}.`);
  }
};

main();

