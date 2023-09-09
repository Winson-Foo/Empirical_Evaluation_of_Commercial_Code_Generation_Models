export default function themePalette(theme) {
  const { colors, customization } = theme;
 
  const primaryColors = {
    light: customization.isDarkMode ? colors?.darkPrimaryLight : colors?.primaryLight,
    main: colors?.primaryMain,
    dark: customization.isDarkMode ? colors?.darkPrimaryDark : colors?.primaryDark,
    200: customization.isDarkMode ? colors?.darkPrimary200 : colors?.primary200,
    800: customization.isDarkMode ? colors?.darkPrimary800 : colors?.primary800
  };
 
  const secondaryColors = {
    light: customization.isDarkMode ? colors?.darkSecondaryLight : colors?.secondaryLight,
    main: customization.isDarkMode ? colors?.darkSecondaryMain : colors?.secondaryMain,
    dark: customization.isDarkMode ? colors?.darkSecondaryDark : colors?.secondaryDark,
    200: colors?.secondary200,
    800: colors?.secondary800
  };
 
  const errorColors = {
    light: colors?.errorLight,
    main: colors?.errorMain,
    dark: colors?.errorDark
  };
 
  const orangeColors = {
    light: colors?.orangeLight,
    main: colors?.orangeMain,
    dark: colors?.orangeDark
  };
 
  const warningColors = {
    light: colors?.warningLight,
    main: colors?.warningMain,
    dark: colors?.warningDark
  };
 
  const successColors = {
    light: colors?.successLight,
    200: colors?.success200,
    main: colors?.successMain,
    dark: colors?.successDark
  };
 
  const greyColors = {
    50: colors?.grey50,
    100: colors?.grey100,
    200: colors?.grey200,
    300: colors?.grey300,
    500: colors.darkTextSecondary,
    600: theme.heading,
    700: colors.darkTextPrimary,
    900: colors.textDark
  };
 
  const darkColors = {
    light: colors?.darkTextPrimary,
    main: colors?.darkLevel1,
    dark: colors?.darkLevel2,
    800: colors?.darkBackground,
    900: colors?.darkPaper
  };
 
  const textColors = {
    primary: colors.darkTextPrimary,
    secondary: colors.darkTextSecondary,
    dark: colors.textDark,
    hint: colors?.grey100
  };
 
  const backgroundColors = {
    paper: theme.paper,
    default: theme.backgroundDefault
  };
 
  const cardColors = {
    main: customization.isDarkMode ? colors?.darkPrimaryMain : colors?.paper,
    light: customization.isDarkMode ? colors?.darkPrimary200 : colors?.paper,
    hover: customization.isDarkMode ? colors?.darkPrimary800 : colors?.paper
  };
 
  const asyncSelectColor = {
    main: customization.isDarkMode ? colors?.darkPrimary800 : colors?.grey50
  };
 
  const canvasHeaderColors = {
    deployLight: colors?.primaryLight,
    deployDark: colors?.primaryDark,
    saveLight: colors?.secondaryLight,
    saveDark: colors?.secondaryDark,
    settingsLight: colors?.grey300,
    settingsDark: colors?.grey700
  };
 
  const codeEditorColors = {
    main: customization.isDarkMode ? colors?.darkPrimary800 : colors?.primaryLight
  };
 
  return {
    mode: customization?.navType,
    common: {
      black: colors?.darkPaper
    },
    primary: primaryColors,
    secondary: secondaryColors,
    error: errorColors,
    orange: orangeColors,
    warning: warningColors,
    success: successColors,
    grey: greyColors,
    dark: darkColors,
    text: textColors,
    background: backgroundColors,
    card: cardColors,
    asyncSelect: asyncSelectColor,
    canvasHeader: canvasHeaderColors,
    codeEditor: codeEditorColors
  };
}

