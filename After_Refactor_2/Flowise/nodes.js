import client from './client';

const NODES_ENDPOINT = '/nodes';

/**
 * Returns all nodes.
 *
 * @returns {Promise} A Promise object representing the http request.
 */
const getAllNodes = () => client.get(NODES_ENDPOINT);

/**
 * Returns a specific node by name.
 *
 * @param   {string}  name  The name of the node to retrieve.
 * @returns {Promise} A Promise object representing the http request.
 */
const getSpecificNode = (name) => client.get(`${NODES_ENDPOINT}/${name}`);

export default {
    getAllNodes,
    getSpecificNode
};