// apiClient.js
import axios from 'axios'
import { API_BASE_URL, API_HEADERS } from 'constants'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: API_HEADERS
})

export default apiClient

// constants.js
export const API_BASE_URL = 'https://www.example.com/api/v1'
export const API_HEADERS = {
  'Content-Type': 'application/json'
}