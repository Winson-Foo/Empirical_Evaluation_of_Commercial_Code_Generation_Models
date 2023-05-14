import client from './client';

const getAllNodes = () => {
  return client.get('/nodes');
};

const getSpecificNode = (name) => {
  return client.get(`/nodes/${name}`);
};

export default {
  getAllNodes,
  getSpecificNode,
};

