import { Dispatch, ReactNode, SetStateAction, createContext, useContext, useState } from 'react'

import { Result } from 'types/common'

interface IResultContext {
  result: Result | null
  setResult: Dispatch<SetStateAction<Result>>
  file: File | null
  setFile: Dispatch<SetStateAction<File | null>>
}

type ResultProviderProps = {
  children: ReactNode
}

const ResultContext = createContext<IResultContext>({} as IResultContext)

export const ResultProvider = ({ children }: ResultProviderProps) => {
  const [result, setResult] = useState<Result | null>(null)
  const [file, setFile] = useState<File | null>(null)

  const value: IResultContext = {
    result,
    setResult,
    file,
    setFile,
  }

  return <ResultContext.Provider value={value}>{children}</ResultContext.Provider>
}

export const useResultContext = () => useContext(ResultContext)
