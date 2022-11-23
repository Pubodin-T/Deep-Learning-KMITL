import { CloseOutlined } from '@mui/icons-material'
import { Box, Button, IconButton, Stack, Typography, styled } from '@mui/material'
import { uploadImage } from 'apis/predict'
import { useResultContext } from 'contexts/ResultContext'
import { isNumber, isString } from 'lodash'
import { useRouter } from 'next/router'
import { ReactNode, useRef, useState } from 'react'
import { UploadImage } from 'views/components/UploadImage'

import { resultTransformer } from 'utils/transformer'

const BannerContainer = styled(Box)(() => ({
  position: 'relative',
  flex: '1 1 0px',
  height: '100%',
}))

const FruitBanner = styled('img')(() => ({
  position: 'absolute',
  bottom: 0,
  right: 0,
  zIndex: -1,
}))

const Fruit1 = styled('img')(() => ({
  position: 'absolute',
  bottom: 80,
  right: 420,
}))

const Fruit2 = styled('img')(() => ({
  position: 'absolute',
  bottom: 310,
  right: 270,
}))

const Fruit3 = styled('img')(() => ({
  position: 'absolute',
  bottom: 440,
  right: 40,
}))

type ContentProps = {
  onClickStart: () => void
}

const Content = ({ onClickStart }: ContentProps) => {
  return (
    <Stack pl={5} gap={4} flex="1 1 0px">
      <Typography variant="h2" color="primary" fontWeight={700}>
        Fruit <br />
        Tracker
      </Typography>
      <Typography variant="h4" color="primary" fontWeight={700}>
        3 Type of fruits
      </Typography>
      <Button variant="contained" color="primary" size="large" sx={{ maxWidth: 160 }} onClick={onClickStart}>
        Start
      </Button>
    </Stack>
  )
}

type HeaderTextProps = {
  children?: ReactNode
}

const HeaderText = ({ children }: HeaderTextProps) => (
  <Stack
    py={2}
    direction="row"
    width="100%"
    alignItems="center"
    justifyContent="center"
    sx={{ backgroundColor: 'white' }}
  >
    <Typography variant="h2" color="secondary" fontWeight={700}>
      {children}
    </Typography>
  </Stack>
)

type CardProps = {
  children?: ReactNode
  imgSrc?: string
  reverse?: boolean
}

const Card = ({ imgSrc, reverse, children }: CardProps) => {
  return (
    <Stack
      direction={reverse ? 'row' : 'row-reverse'}
      width="100%"
      alignItems="center"
      justifyContent="center"
      height={360}
    >
      <Stack
        py={2}
        direction="row"
        flex={1}
        alignItems="center"
        justifyContent="center"
        sx={{ backgroundColor: 'background.default' }}
      >
        {isString(children) || isNumber(children) ? (
          <Typography variant="h2" color="secondary" fontWeight={700} textTransform="capitalize">
            {children}
          </Typography>
        ) : (
          children
        )}
      </Stack>
      <Stack alignSelf="stretch" flex={1} alignItems="center" justifyContent="center" sx={{ backgroundColor: 'white' }}>
        <img src={imgSrc} width={300} />
      </Stack>
    </Stack>
  )
}

const Banner = () => (
  <BannerContainer>
    <Fruit1 src="/static/images/fruit-1.svg" width={200} />
    <Fruit2 src="/static/images/fruit-2.svg" width={200} />
    <Fruit3 src="/static/images/fruit-3.svg" width={200} />
    <FruitBanner src="/static/images/fruit-banner.svg" width={540} />
  </BannerContainer>
)

const data: Array<Omit<CardProps, 'children' | 'reverse'> & { text: string }> = [
  {
    imgSrc: '/static/images/longgong.svg',
    text: 'longgong',
  },
  {
    imgSrc: '/static/images/langsat.svg',
    text: 'langsat',
  },
  {
    imgSrc: '/static/images/longan.svg',
    text: 'longan',
  },
]

export const Home = () => {
  const { setResult, file, setFile } = useResultContext()

  const router = useRouter()

  const trackerRef = useRef<HTMLDivElement | null>(null)

  const handlePredict = async () => {
    if (!file) return

    const formData = new FormData()
    formData.append('file', file)

    try {
      const result = await uploadImage(formData)
      setResult(resultTransformer(result))

      router.push('/result')
    } catch (err) {
      console.log(err)
    }
  }

  return (
    <Box>
      <Stack pl={8} direction="row" alignItems="center" justifyContent="center" sx={{ height: 'calc(100vh - 80px)' }}>
        <Content onClickStart={() => trackerRef.current?.scrollIntoView()} />
        <Banner />
      </Stack>
      <HeaderText>Fruits</HeaderText>
      {data.map(({ text, imgSrc }, i) => (
        <Card key={text} imgSrc={imgSrc} reverse={!!(i & 1)}>
          {text}
        </Card>
      ))}
      <HeaderText>Tracker</HeaderText>
      <Card imgSrc="/static/images/upload.svg">
        <Stack gap={2} alignItems="center" ref={trackerRef}>
          <Typography variant="h3" color="primary" fontWeight={700}>
            Fruits Tracker
          </Typography>
          <Typography variant="h6" color="black" fontWeight={700}>
            Upload your image for Tracker
          </Typography>
          {file?.name ? (
            <>
              <Stack direction="row" gap={2} alignItems="center">
                <Typography variant="body2" color="primary">
                  {file.name}
                </Typography>
                <IconButton onClick={() => setFile(null)}>
                  <CloseOutlined width={16} />
                </IconButton>
              </Stack>
              <Button variant="contained" color="secondary" size="large" sx={{ minWidth: 144 }} onClick={handlePredict}>
                Predict
              </Button>
            </>
          ) : (
            <UploadImage setFile={(file) => setFile(file)} />
          )}
        </Stack>
      </Card>
    </Box>
  )
}
