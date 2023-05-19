import client from './client';

// Returns a list of all available marketplaces
const fetchAllMarketplaces = async () => {
  try {
    const response = await client.get('/marketplaces');
    return response.data;
  } catch (error) {
    console.error(`Error fetching marketplaces: ${error.message}`);
  }
};

export default {
  fetchAllMarketplaces,
};

