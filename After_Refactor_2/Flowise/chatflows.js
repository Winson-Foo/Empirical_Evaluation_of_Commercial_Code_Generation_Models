import client from './client';

const ENDPOINT_CHATFLOWS = '/chatflows'; // common endpoint for all chatflow-related requests

/**
 * Fetch all chatflows
 * @return {Promise} Promise object representing the result of the API call
 */
const getAllChatflows = () => client.get(ENDPOINT_CHATFLOWS);

/**
 * Fetch a specific chatflow by ID
 * @param  {string} id ID of the chatflow to retrieve
 * @return {Promise}   Promise object representing the result of the API call
 */
const getSpecificChatflow = (id) => client.get(`${ENDPOINT_CHATFLOWS}/${id}`);

/**
 * Create a new chatflow
 * @param  {object} body The request body to send
 * @return {Promise}     Promise object representing the result of the API call
 */
const createNewChatflow = (body) => client.post(ENDPOINT_CHATFLOWS, body);

/**
 * Update an existing chatflow by ID
 * @param  {string} id   ID of the chatflow to update
 * @param  {object} body The request body to send
 * @return {Promise}     Promise object representing the result of the API call
 */
const updateChatflow = (id, body) => client.put(`${ENDPOINT_CHATFLOWS}/${id}`, body);

/**
 * Delete a chatflow by ID
 * @param  {string} id ID of the chatflow to delete
 * @return {Promise}   Promise object representing the result of the API call
 */
const deleteChatflow = (id) => client.delete(`${ENDPOINT_CHATFLOWS}/${id}`);

export default {
    getAllChatflows,
    getSpecificChatflow,
    createNewChatflow,
    updateChatflow,
    deleteChatflow
};

