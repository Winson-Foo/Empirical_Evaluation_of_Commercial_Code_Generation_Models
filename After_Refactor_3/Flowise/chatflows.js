import client from './client';

// Constants
const CHATFLOWS_ENDPOINT = '/chatflows';
const ID_PROPERTY = 'id';
const BODY_PROPERTY = 'body';

// Returns all chatflows
// Input: None
// Output: Promise that resolves to an array of chatflow objects
const getAllChatflows = () => client.get(CHATFLOWS_ENDPOINT);

// Returns a specific chatflow with the given id
// Input: Chatflow id (integer)
// Output: Promise that resolves to a chatflow object
const getSpecificChatflow = (id) => client.get(`${CHATFLOWS_ENDPOINT}/${id}`);

// Creates a new chatflow with the given body
// Input: Chatflow object
// Output: Promise that resolves to the created chatflow object
const createNewChatflow = (body) => client.post(CHATFLOWS_ENDPOINT, {[BODY_PROPERTY]: body});

// Updates a specific chatflow with the given id and body
// Input: Chatflow id (integer) and chatflow object
// Output: Promise that resolves to the updated chatflow object
const updateChatflow = (id, body) => client.put(`${CHATFLOWS_ENDPOINT}/${id}`, {[BODY_PROPERTY]: body});

// Deletes a specific chatflow with the given id
// Input: Chatflow id (integer)
// Output: Promise that resolves with no content
const deleteChatflow = (id) => client.delete(`${CHATFLOWS_ENDPOINT}/${id}`);

// Export functions
export default {
    getAllChatflows,
    getSpecificChatflow,
    createNewChatflow,
    updateChatflow,
    deleteChatflow
};

