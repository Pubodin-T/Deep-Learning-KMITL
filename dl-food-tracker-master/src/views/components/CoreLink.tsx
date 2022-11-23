import { Box, Link as MuiLink, styled } from '@mui/material'
import Link from 'next/link'
import { ReactNode } from 'react'

const shouldForwardProp = (props: string) => !['isLinkStyle'].includes(props)
const InternalLinkContainer = styled(Box, { shouldForwardProp })<{ isLinkStyle?: boolean }>(
  ({ theme, isLinkStyle }) => ({
    cursor: 'pointer',
    display: 'contents',
    color: isLinkStyle ? theme.palette.link : 'unset',
  }),
)

type CoreLinkProps = {
  href?: string
  path?: string
  children: ReactNode
  download?: string
  className?: string
  isLinkStyle?: boolean
}

export const CoreLink = ({ href, path, children, download, className, isLinkStyle }: CoreLinkProps) => {
  if (path) {
    return (
      <Link href={path} passHref>
        <InternalLinkContainer isLinkStyle={isLinkStyle} className={className}>
          {children}
        </InternalLinkContainer>
      </Link>
    )
  }

  if (href) {
    return (
      <MuiLink
        className={className}
        href={href}
        target="_blank"
        rel="noopenner noreferrer"
        sx={{ display: 'contents' }}
        download={download}
      >
        {children}
      </MuiLink>
    )
  }

  return <>{children}</>
}
