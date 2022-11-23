import { createTheme } from '@mui/material/styles'

const NAVBAR_HEIGHT = 0

const theme = createTheme({
  breakpoints: {
    values: {
      xs: 0,
      sm: 600,
      md: 960,
      lg: 1280,
      xl: 1920,
    },
  },
  palette: {
    primary: {
      light: '#D7C0AE',
      main: '#967E76',
    },
    secondary: {
      light: '#B7C4CF',
      main: '#6B7280',
      dark: '#545964',
    },
    background: {
      default: '#F5EFE6',
    },
    link: '#967E76',
    //   text: {},
    //   background: {},
    //   success: {},
    //   warning: {},
    //   error: {},
    //   info: {},
  },
  typography: {
    fontFamily: `Lexend, sans-serif`,
    h1: { fontSize: '5rem', lineHeight: '6rem', fontWeight: 'normal', letterSpacing: 0 },
    h2: { fontSize: '3.75rem', lineHeight: '4.5rem', fontWeight: 'normal', letterSpacing: 0 },
    h3: { fontSize: '3rem', lineHeight: '3.625rem', fontWeight: 'normal', letterSpacing: 0 },
    h4: { fontSize: '2.5rem', lineHeight: '3rem', fontWeight: 'normal', letterSpacing: 0 },
    h5: { fontSize: '2rem', lineHeight: '2.5rem', fontWeight: 'normal', letterSpacing: 0 },
    h6: { fontSize: '1.5rem', lineHeight: '2.25rem', fontWeight: 'normal', letterSpacing: 0 },
    subtitle1: { fontSize: '1.25rem', lineHeight: '1.75rem', fontWeight: 'lighter', letterSpacing: 0 },
    subtitle2: { fontSize: '1.2rem', fontWeight: 'lighter' },
    body1: { fontSize: '1.125rem', lineHeight: '1.75rem', fontWeight: 'normal', letterSpacing: 0 },
    body2: { fontSize: '1rem', lineHeight: '1.625rem', fontWeight: 'normal', letterSpacing: 0 },
    label: { fontSize: '0.875rem', lineHeight: '1.375rem', fontWeight: 'normal', letterSpacing: 0 },
    caption: { fontSize: '0.75rem', lineHeight: '1.125rem', fontWeight: 'normal', letterSpacing: 0 },
  },
  shadows: [
    'none',
    '4px 4px 4px 0 rgba(0,0,0,0.2), -4px -4px 4px 0 rgba(255,255,255,0.04)',
    'inset 4px 4px 4px 0 rgba(0,0,0,0.2), inset -4px -4px 4px 0 rgba(255,255,255,0.04)',
    '8px 8px 8px 0 rgba(0,0,0,0.2), -8px -8px 8px 0 rgba(255,255,255,0.04)',
    'inset 8px 8px 8px 0 rgba(0,0,0,0.2), inset -8px -8px 8px 0 rgba(255,255,255,0.04)',
    '0px 3px 5px -1px rgba(0,0,0,0.1),0px 5px 8px 0px rgba(0,0,0,0.04),0px 1px 14px 0px rgba(0,0,0,0.02)',
    '0px 3px 5px -1px rgba(0,0,0,0.1),0px 6px 10px 0px rgba(0,0,0,0.04),0px 1px 18px 0px rgba(0,0,0,0.02)',
    '0px 4px 5px -2px rgba(0,0,0,0.1),0px 7px 10px 1px rgba(0,0,0,0.04),0px 2px 16px 1px rgba(0,0,0,0.02)',
    '0px 5px 5px -3px rgba(0,0,0,0.1),0px 8px 10px 1px rgba(0,0,0,0.04),0px 3px 14px 2px rgba(0,0,0,0.02)',
    '0px 5px 6px -3px rgba(0,0,0,0.1),0px 9px 12px 1px rgba(0,0,0,0.04),0px 3px 16px 2px rgba(0,0,0,0.02)',
    '0px 6px 6px -3px rgba(0,0,0,0.1),0px 10px 14px 1px rgba(0,0,0,0.04),0px 4px 18px 3px rgba(0,0,0,0.02)',
    '0px 6px 7px -4px rgba(0,0,0,0.1),0px 11px 15px 1px rgba(0,0,0,0.04),0px 4px 20px 3px rgba(0,0,0,0.02)',
    '0px 7px 8px -4px rgba(0,0,0,0.1),0px 12px 17px 2px rgba(0,0,0,0.04),0px 5px 22px 4px rgba(0,0,0,0.02)',
    '0px 7px 8px -4px rgba(0,0,0,0.1),0px 13px 19px 2px rgba(0,0,0,0.04),0px 5px 24px 4px rgba(0,0,0,0.02)',
    '0px 7px 9px -4px rgba(0,0,0,0.1),0px 14px 21px 2px rgba(0,0,0,0.04),0px 5px 26px 4px rgba(0,0,0,0.02)',
    '0px 8px 9px -5px rgba(0,0,0,0.1),0px 15px 22px 2px rgba(0,0,0,0.04),0px 6px 28px 5px rgba(0,0,0,0.02)',
    '0px 8px 10px -5px rgba(0,0,0,0.1),0px 16px 24px 2px rgba(0,0,0,0.04),0px 6px 30px 5px rgba(0,0,0,0.02)',
    '0px 8px 11px -5px rgba(0,0,0,0.1),0px 17px 26px 2px rgba(0,0,0,0.04),0px 6px 32px 5px rgba(0,0,0,0.02)',
    '0px 9px 11px -5px rgba(0,0,0,0.1),0px 18px 28px 2px rgba(0,0,0,0.04),0px 7px 34px 6px rgba(0,0,0,0.02)',
    '0px 9px 12px -6px rgba(0,0,0,0.1),0px 19px 29px 2px rgba(0,0,0,0.04),0px 7px 36px 6px rgba(0,0,0,0.02)',
    '0px 10px 13px -6px rgba(0,0,0,0.1),0px 20px 31px 3px rgba(0,0,0,0.04),0px 8px 38px 7px rgba(0,0,0,0.02)',
    '0px 10px 13px -6px rgba(0,0,0,0.1),0px 21px 33px 3px rgba(0,0,0,0.04),0px 8px 40px 7px rgba(0,0,0,0.02)',
    '0px 10px 14px -6px rgba(0,0,0,0.1),0px 22px 35px 3px rgba(0,0,0,0.04),0px 8px 42px 7px rgba(0,0,0,0.02)',
    '0px 11px 14px -7px rgba(0,0,0,0.1),0px 23px 36px 3px rgba(0,0,0,0.04),0px 9px 44px 8px rgba(0,0,0,0.02)',
    '0px 11px 15px -7px rgba(0,0,0,0.1),0px 24px 38px 3px rgba(0,0,0,0.04),0px 9px 46px 8px rgba(0,0,0,0.02)',
  ],
  shape: {
    borderRadius: 4,
  },
})

theme.components = {
  MuiCssBaseline: {
    styleOverrides: {
      html: {
        scrollBehavior: 'smooth',
        scrollPaddingTop: `${NAVBAR_HEIGHT}px`, // * For the Navbar height
      },
      a: {
        color: 'inherit',
        textDecoration: 'none',
      },
      body: {
        backgroundColor: theme.palette.background.default,
      },
    },
  },
  MuiButton: {
    styleOverrides: {
      root: {
        textTransform: 'none',
        fontWeight: 700,
        boxShadow: 'unset',
        borderRadius: 12,
        fontSize: '1.125rem',
      },
      containedPrimary: {
        backgroundColor: theme.palette.primary.light,
        color: theme.palette.primary.main,
        '&:hover': {
          boxShadow: 'unset',
          backgroundColor: theme.palette.primary.main,
          color: theme.palette.common.white,
        },
      },
      containedSecondary: {
        backgroundColor: theme.palette.secondary.main,
        color: theme.palette.common.white,
        '&:hover': {
          boxShadow: 'unset',
          backgroundColor: theme.palette.secondary.dark,
          color: theme.palette.common.white,
        },
      },
    },
  },
}

export { theme }
