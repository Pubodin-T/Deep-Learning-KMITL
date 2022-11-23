import axios from 'axios'

import { TRACKER_BACKEND_URL } from 'constants/constants'

const httpClient = axios.create({
  baseURL: TRACKER_BACKEND_URL,
  timeout: 5000,
})

export { httpClient }
