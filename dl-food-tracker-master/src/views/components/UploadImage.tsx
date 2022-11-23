import { Button, Stack } from '@mui/material'
import { ChangeEvent, ChangeEventHandler, useRef } from 'react'

type UploadImageProps = {
  setFile: (image: File) => void
}

export const UploadImage = ({ setFile }: UploadImageProps) => {
  const inputRef = useRef<HTMLInputElement | null>(null)

  const onUpload: ChangeEventHandler<HTMLInputElement> = async (event: ChangeEvent<HTMLInputElement>) => {
    if (!event.target?.files || !event.target.files?.[0]) return
    setFile(event.target.files[0])
  }

  return (
    <>
      <input ref={inputRef} accept="image/*" type="file" onChange={onUpload} hidden />
      <Button variant="contained" color="secondary" size="large" onClick={() => inputRef.current?.click()}>
        <Stack alignItems="center" justifyContent="center" width={100}>
          <img src="/static/icons/upload.svg" width={32} />
        </Stack>
      </Button>
    </>
  )
}
