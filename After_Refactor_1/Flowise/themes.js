// themes.js

const defaultTheme = (customization) => {
    return {
      palette: {
        primary: {
          main: customization.primary,
        },
        secondary: {
          main: customization.secondary,
        },
      },
    }
  }
  
  export { defaultTheme }