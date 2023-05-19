// import the client module for HTTP requests
import client from './clientModule';

/**
 * Gets all marketplaces from the server.
 * @returns {Promise} A promise representing the HTTP request.
 */
const getMarketplaces = () => {
    return client.get('/marketplaces');
}

// export the function for reuse in other modules
export default {
    getMarketplaces
} 

// In clientModule.js

import axios from 'axios';

// create an axios instance for HTTP requests
const client = axios.create({
    baseURL: 'https://example.com/api'
});

// export the client instance for use in other modules
export default client;

