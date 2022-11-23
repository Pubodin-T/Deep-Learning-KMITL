import { Result } from 'types/common'
import { ResultResponse } from 'types/dto'

export const resultTransformer = (resultResponse: ResultResponse): Result => {
  return {
    langsat: resultResponse.langsat ?? 0,
    longan: resultResponse.longan ?? 0,
    longgong: resultResponse.longgong ?? 0,
  }
}
