import apiClient from './client';

const sendPredictionRequest = (id, input) => {
  return apiClient.post(`/internal-prediction/${id}`, input)
         .then(response => response.data)
         .catch(error => {
           console.error(`Error in sending prediction request: ${error}`);
           throw error;
         });
};

export default {
  sendPredictionRequest
} 