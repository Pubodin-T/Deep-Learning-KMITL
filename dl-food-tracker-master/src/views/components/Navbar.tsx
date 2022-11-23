import { AppBar, Box, Button, Stack, Toolbar, styled } from '@mui/material'
import { CoreLink } from 'views/components/CoreLink'

const StyledAppBar = styled(AppBar)(({ theme }) => ({
  boxShadow: 'unset',
  backgroundColor: theme.palette.background.default,
  padding: theme.spacing(0, 5),
}))

const StyledButton = styled(Button)(() => ({
  fontSize: '24px',
}))

export const Navbar = () => {
  return (
    <StyledAppBar position="relative">
      <Box py={1}>
        <Toolbar disableGutters>
          <Stack direction="row" width="100%" justifyContent="center" alignItems="center" position="relative" gap={4}>
            <Stack position="absolute" left={0} justifyContent="center" alignItems="center">
              <CoreLink path="/">
                <img src="/static/images/logo.svg" width={180} />
              </CoreLink>
            </Stack>
            <CoreLink path="/">
              <StyledButton variant="text" color="secondary">
                Home
              </StyledButton>
            </CoreLink>
            <CoreLink path="/result">
              <StyledButton variant="text" color="secondary">
                Tracker
              </StyledButton>
            </CoreLink>
            <Stack position="absolute" right={0} justifyContent="center" alignItems="center">
              <CoreLink path="/">
                <img src="/static/images/question.svg" width={48} />
              </CoreLink>
            </Stack>
          </Stack>
        </Toolbar>
      </Box>
    </StyledAppBar>
  )
}
