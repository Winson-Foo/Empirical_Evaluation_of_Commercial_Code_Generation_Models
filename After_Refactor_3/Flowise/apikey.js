import client from './client'

export const getAPIKeys = () => client.get('/apikey')
export const createAPI = (body) => client.post('/apikey', body)
export const updateAPI = (id, body) => client.put(`/apikey/${id}`, body)
export const deleteAPI = (id) => client.delete(`/apikey/${id}`)

import { getAPIKeys, createAPI, updateAPI, deleteAPI } from './api'

export {
  getAPIKeys,
  createAPI,
  updateAPI,
  deleteAPI
}