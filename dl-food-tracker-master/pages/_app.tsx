import { CacheProvider, EmotionCache, ThemeProvider as EmotionThemeProvider } from '@emotion/react'
import { CssBaseline, ThemeProvider as MaterialThemeProvider, StyledEngineProvider } from '@mui/material'
import { ResultProvider } from 'contexts/ResultContext'
import { AppProps } from 'next/app'
import Head from 'next/head'
import { PageWrapper } from 'views/components/PageWrapper'

import { createEmotionCache } from 'libs/emotion'

import { theme } from 'themes/main'

interface CustomAppProps extends AppProps {
  emotionCache?: EmotionCache
}

const clientSideEmotionCache = createEmotionCache()

const CustomApp = ({ Component, emotionCache = clientSideEmotionCache, pageProps }: CustomAppProps) => {
  return (
    <>
      <Head>
        <title>DL Food Tracker</title>
        <meta name="viewport" content="initial-scale=1, width=device-width" />
      </Head>
      <StyledEngineProvider injectFirst>
        <CacheProvider value={emotionCache}>
          <MaterialThemeProvider theme={theme}>
            <EmotionThemeProvider theme={theme}>
              <CssBaseline />
              <ResultProvider>
                <PageWrapper>
                  <Component {...pageProps} />
                </PageWrapper>
              </ResultProvider>
            </EmotionThemeProvider>
          </MaterialThemeProvider>
        </CacheProvider>
      </StyledEngineProvider>
    </>
  )
}

export default CustomApp
