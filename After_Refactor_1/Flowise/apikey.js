import client from './client';

const API_KEY_PATH = '/apikey';

const getAllAPIKeys = async () => {
  try {
    const response = await client.get(API_KEY_PATH);
    return response.data;
  } catch (error) {
    console.error(error);
    throw error;
  }
};

const createNewAPI = async body => {
  try {
    const response = await client.post(API_KEY_PATH, body);
    return response.data;
  } catch (error) {
    console.error(error);
    throw error;
  }
};

const updateAPI = async (id, body) => {
  try {
    const response = await client.put(`${API_KEY_PATH}/${id}`, body);
    return response.data;
  } catch (error) {
    console.error(error);
    throw error;
  }
};

const deleteAPI = async id => {
  try {
    const response = await client.delete(`${API_KEY_PATH}/${id}`);
    return response.data;
  } catch (error) {
    console.error(error);
    throw error;
  }
};

export { getAllAPIKeys, createNewAPI, updateAPI, deleteAPI };

