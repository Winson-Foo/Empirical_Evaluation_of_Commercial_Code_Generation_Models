/* eslint-disable no-console */
require('dotenv').config();
const fs = require('fs');
const path = require('path');

function updateTrackerEndpoint(endPoint) {
  const file = path.resolve(__dirname, '../public/script.js');
  const tracker = fs.readFileSync(file);
  fs.writeFileSync(
    file,
    tracker.toString().replace(/"\/api\/send"/g, `"${endPoint}"`),
  );
  console.log(`Updated tracker endpoint: ${endPoint}.`);
}

const endPoint = process.env.COLLECT_API_ENDPOINT;
if (endPoint) {
  updateTrackerEndpoint(endPoint);
} else {
  console.error('Collect API endpoint is not defined.');
}

