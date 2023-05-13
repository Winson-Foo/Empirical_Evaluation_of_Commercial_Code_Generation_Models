import client from './client';

// Function to get all API keys from the server
// Returns: Promise containing the response data
// Throws: Error if the request fails
const getAllAPIKeys = () => {
  try {
    return client.get('/apikey');
  } catch (err) {
    throw new Error(`Failed to get all API keys: ${err.message}`);
  }
};

// Function to create a new API key
// Parameters:
//   - body: Object containing the API key data
// Returns: Promise containing the response data
// Throws: Error if the request fails
const createNewAPI = (body) => {
  try {
    return client.post(`/apikey`, body);
  } catch (err) {
    throw new Error(`Failed to create new API: ${err.message}`);
  }
};

// Function to update an existing API key
// Parameters:
//   - id: String containing the ID of the API key to update
//   - body: Object containing the updated API key data
// Returns: Promise containing the response data
// Throws: Error if the request fails
const updateAPI = (id, body) => {
  try {
    return client.put(`/apikey/${id}`, body);
  } catch (err) {
    throw new Error(`Failed to update API key ${id}: ${err.message}`);
  }
};

// Function to delete an existing API key
// Parameters:
//   - id: String containing the ID of the API key to delete
// Returns: Promise containing the response data
// Throws: Error if the request fails
const deleteAPI = (id) => {
  try {
    return client.delete(`/apikey/${id}`);
  } catch (err) {
    throw new Error(`Failed to delete API key ${id}: ${err.message}`);
  }
};

// Export all functions as an object
export default {
  getAllAPIKeys,
  createNewAPI,
  updateAPI,
  deleteAPI,
};

