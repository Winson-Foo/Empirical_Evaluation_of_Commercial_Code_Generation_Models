import client from './client';

const chatflowAPI = {
  // Get all chatflows from the server
  getAll: () => client.get('/chatflows'),

  // Get a specific chatflow by ID from the server
  get: (id) => client.get(`/chatflows/${id}`),

  // Create a new chatflow on the server
  create: (data) => client.post('/chatflows', data),

  // Update an existing chatflow on the server
  update: (id, data) => client.put(`/chatflows/${id}`, data),

  // Delete a chatflow from the server
  delete: (id) => client.delete(`/chatflows/${id}`),
};

export default chatflowAPI;

