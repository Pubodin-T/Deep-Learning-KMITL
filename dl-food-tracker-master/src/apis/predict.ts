import { ResultResponse } from 'types/dto'

import { httpClient } from './httpClient'

export const uploadImage = async (data: FormData): Promise<ResultResponse> => {
  return httpClient
    .post<ResultResponse>('/predict/image', data, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    .then((response) => {
      if (response.status === 200) {
        return response.data
      }

      throw new Error(`[uploadImage] response with status ${response.status}`)
    })
    .catch((err) => {
      console.error('[uploadImage] API Error')

      throw new Error(err.response?.data?.message ?? err.message)
    })
}
