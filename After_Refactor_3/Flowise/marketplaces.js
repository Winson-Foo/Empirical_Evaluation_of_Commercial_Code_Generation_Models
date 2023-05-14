import api from './api';

// Fetches all marketplaces from server
const getAllMarketplaces = async () => {
  try {
    const response = await api.get('/marketplaces');
    return response.data;
  } catch (error) {
    console.error(error);
  }
};

export default {
  getAllMarketplaces,
};

