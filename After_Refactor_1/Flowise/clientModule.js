import axios from 'axios';

// create an axios instance for HTTP requests
const client = axios.create({
    baseURL: 'https://example.com/api'
});

// export the client instance for use in other modules
export default client;