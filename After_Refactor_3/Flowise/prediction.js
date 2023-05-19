import client from './client';

/**
 * Calls the internal prediction endpoint and retrieves the prediction results.
 *
 * @param {string} id - Identifier for the prediction service.
 * @param {object} input - Input data for the prediction service.
 *
 * @return {Promise} - Promise that resolves with the prediction results.
 */
const sendMessageAndGetPrediction = async (id, input) => {
  try {
    const response = await client.post(`/internal-prediction/${id}`, input);
    return response.data;
  } catch (error) {
    console.error(error);
    throw new Error('Failed to retrieve prediction results.');
  }
};

export default {
  sendMessageAndGetPrediction,
};

