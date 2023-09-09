import client from './client'

/**
 * Returns all nodes
 * @returns {Promise}
 */
const getAllNodes = () => client.get('/nodes')

/**
 * Returns a specific node by name
 * @param {string} name
 * @returns {Promise}
 */
const getSpecificNode = (name) => client.get(`/nodes/${name}`)

export default {
    getAllNodes,
    getSpecificNode
}