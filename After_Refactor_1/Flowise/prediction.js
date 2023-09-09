import client from './client';

/**
 * Sends a message and gets prediction from the server.
 *
 * @param {String} id - The id of the prediction.
 * @param {Object} input - The input for the prediction.
 * @returns {Promise} A promise that resolves with the prediction.
 */
const sendMessageAndGetPrediction = (id, input) => {
  return client.post(`/internal-prediction/${id}`, input);
};

export default { sendMessageAndGetPrediction };

