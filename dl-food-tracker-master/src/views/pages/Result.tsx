import { Box, Stack, Typography, alpha, styled } from '@mui/material'
import { useResultContext } from 'contexts/ResultContext'
import { useMemo } from 'react'

const CIRCLE_SCALE = 900

const Root = styled(Box)(({ theme }) => ({
  padding: theme.spacing(6),
  position: 'relative',
  backgroundColor: 'white',
  height: 'calc(100vh - 80px)',
  overflow: 'hidden',
}))

const ContentWrapper = styled(Box)(({ theme }) => ({
  padding: theme.spacing(8),
  border: `2px solid ${theme.palette.primary.main}`,
  position: 'relative',
  height: '100%',
  zIndex: 2,
}))

const BorderCircle = styled(Stack)(({ theme }) => ({
  padding: theme.spacing(4),
  justifyContent: 'center',
  alignItems: 'center',
  border: `2px solid ${alpha(theme.palette.common.black, 0.2)}`,
  borderRadius: '50%',
  width: 200,
  height: 200,
}))

const CircleClip = styled(Box)(({ theme }) => ({
  borderRadius: '100%',
  width: 160,
  height: 160,
  backgroundColor: theme.palette.common.white,
  overflow: 'hidden',
}))

const Crown = styled(Box)(() => ({
  borderRadius: '100%',
  position: 'absolute',
  width: 300,
  height: 300,
  marginTop: -80,
}))

const SecondaryCircle = styled(Box)(({ theme }) => ({
  width: CIRCLE_SCALE,
  height: CIRCLE_SCALE,
  zIndex: 1,
  backgroundColor: theme.palette.secondary.light,
  position: 'absolute',
  bottom: -CIRCLE_SCALE / 1.3,
  borderRadius: '50%',
  left: '50%',
  transform: 'translateX(-50%)',
}))

const HeaderText = styled(Box)(({ theme }) => ({
  position: 'absolute',
  padding: theme.spacing(0, 4),
  top: -38,
  left: '50%',
  transform: 'translateX(-50%)',
  backgroundColor: theme.palette.common.white,
}))

const ResultContent = styled(Stack)(({ theme }) => ({
  position: 'absolute',
  zIndex: 2,
  alignItems: 'center',
  gap: theme.spacing(2),
  bottom: 0,
  padding: theme.spacing(0, 2, 8),
  left: '50%',
  transform: 'translateX(-50%)',
}))

type FruitPredictCardProps = {
  text: string
  imgSrc?: string
  value?: number
}

const FruitPredictCard = ({ imgSrc, value, text }: FruitPredictCardProps) => {
  if (value === undefined) {
    return null
  }

  return (
    <Stack alignItems="center" gap={3}>
      <Typography variant="h4" color="primary" fontWeight={700}>
        {text}
      </Typography>
      <BorderCircle>
        <img src={imgSrc} width="100%" />
      </BorderCircle>
      <Typography variant="h5" color="primary" fontWeight={700}>
        {(value * 100).toFixed(2)}%
      </Typography>
    </Stack>
  )
}

export const Result = () => {
  const { result, file } = useResultContext()

  const winner: string | undefined = useMemo(() => {
    if (!result) return
    console.log(result)

    return Object.keys(result).reduce((a, b) => (result[a] > result[b] ? a : b))
  }, [result])

  const displayResult = useMemo(() => {
    if (!result) {
      return
    }

    return {
      longgong: +result.longgong.toFixed(4),
      langsat: +result.langsat.toFixed(4),
      longan: +result.longan.toFixed(4),
    }
  }, [result])

  if (!displayResult || !result) {
    return (
      <Stack height="100%" justifyContent="center" alignItems="center">
        <Typography variant="h1" color="error">
          No result
        </Typography>
      </Stack>
    )
  }

  return (
    <Root>
      <ContentWrapper>
        <HeaderText>
          <Typography variant="h2" color="secondary" fontWeight={700}>
            Result
          </Typography>
        </HeaderText>
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <FruitPredictCard text="Langsat" imgSrc="/static/images/langsat.svg" value={displayResult.langsat} />
          <FruitPredictCard text="Longgong" imgSrc="/static/images/longgong.svg" value={displayResult.longgong} />
          <FruitPredictCard text="Longan" imgSrc="/static/images/longan.svg" value={displayResult.longan} />
        </Stack>
      </ContentWrapper>
      <ResultContent>
        <Crown>
          <img src="/static/images/crown.svg" width="100%" height="100%" />
        </Crown>
        <CircleClip>{file && <img src={URL.createObjectURL(file)} width="100%" height="100%" />}</CircleClip>
        <Box width="100%" position="relative" sx={{ zIndex: 5 }}>
          <Typography variant="h3" color="white" fontWeight={700} textAlign="center" textTransform="capitalize">
            {winner}
          </Typography>
        </Box>
      </ResultContent>
      <SecondaryCircle />
    </Root>
  )
}
