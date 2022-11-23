export declare module '@mui/material/styles/createPalette' {
  interface Palette {
    link: string
  }

  interface PaletteOptions {
    link: string
  }
}

export declare module '@mui/material/styles' {
  interface TypographyVariants {
    label: React.CSSProperties
  }

  // * ================================================================================
  // *  allow configuration using `createTheme`
  // * ================================================================================
  interface TypographyVariantsOptions {
    label?: React.CSSProperties
  }
}

// * ================================================================================
// *  Update the Typography's variant prop options
// * ================================================================================
export declare module '@mui/material/Typography' {
  interface TypographyPropsVariantOverrides {
    label: true
  }
}

// * ================================================================================
// *  Update the Button's variant prop options
// * ================================================================================
export declare module '@mui/material/Button' {
  interface ButtonPropsVariantOverrides {
    circle: true
  }
}
