// apiConfig.js
import axios from 'axios';
import { baseURL } from 'store/constant';

const apiClient = axios.create({
  baseURL: `${baseURL}/api/v1`,
  headers: {
    'Content-type': 'application/json',
  },
});

export default apiClient;


// apiCalls.js

import apiClient from './apiConfig';

export const getUsers = () => {
  return apiClient.get('/users');
};

export const getUserById = (userId) => {
  return apiClient.get(`/users/${userId}`);
};

export const createUser = (userData) => {
  return apiClient.post('/users', userData);
};

// usage example
import { getUsers } from './apiCalls';

const users = await getUsers();
console.log(users.data);