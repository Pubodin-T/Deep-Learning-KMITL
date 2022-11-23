// import { useRouter } from 'next/router'
import { Box, Stack, styled } from '@mui/material'
import { ReactNode } from 'react'
import { Navbar } from 'views/components/Navbar'

const Screen = styled(Stack)(() => ({
  flexGrow: 1,
  minHeight: '100vh',
}))

const ContentWrapper = styled(Box)(() => ({
  flex: '1 0 0px',
}))

type PageWrapperProps = {
  children: ReactNode
}

export const PageWrapper = ({ children }: PageWrapperProps) => {
  return (
    <Screen>
      <Navbar />
      <ContentWrapper>{children}</ContentWrapper>
    </Screen>
  )
}
