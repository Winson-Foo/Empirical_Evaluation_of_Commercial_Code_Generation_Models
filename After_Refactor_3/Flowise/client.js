import axios from 'axios';
import { API_VERSION, BASE_URL } from 'store/constants';

const API_PATH = '/api/v1'; // Separating out the API path

const client = axios.create({
  baseURL: `${BASE_URL}${API_PATH}`,
  headers: {
    'Content-Type': 'application/json',
  },
});

export default client;

