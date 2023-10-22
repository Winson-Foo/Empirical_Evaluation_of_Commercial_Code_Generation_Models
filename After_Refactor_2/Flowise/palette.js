export default function themePalette(theme) {
  const { 
    colors,
    customization: { isDarkMode, navType },
  } = theme || {};

  const black = colors?.darkPaper;
  const primaryLight = isDarkMode ? colors?.darkPrimaryLight : colors?.primaryLight;
  const primaryMain = colors?.primaryMain;
  const primaryDark = isDarkMode ? colors?.darkPrimaryDark : colors?.primaryDark;
  const primary200 = isDarkMode ? colors?.darkPrimary200 : colors?.primary200;
  const primary800 = isDarkMode ? colors?.darkPrimary800 : colors?.primary800;
  const secondaryLight = isDarkMode ? colors?.darkSecondaryLight : colors?.secondaryLight;
  const secondaryMain = isDarkMode ? colors?.darkSecondaryMain : colors?.secondaryMain;
  const secondaryDark = isDarkMode ? colors?.darkSecondaryDark : colors?.secondaryDark;
  const secondary200 = colors?.secondary200;
  const secondary800 = colors?.secondary800;
  const errorLight = colors?.errorLight;
  const errorMain = colors?.errorMain;
  const errorDark = colors?.errorDark;
  const orangeLight = colors?.orangeLight;
  const orangeMain = colors?.orangeMain;
  const orangeDark = colors?.orangeDark;
  const warningLight = colors?.warningLight;
  const warningMain = colors?.warningMain;
  const warningDark = colors?.warningDark;
  const successLight = colors?.successLight;
  const success200 = colors?.success200;
  const successMain = colors?.successMain;
  const successDark = colors?.successDark;
  const grey50 = colors?.grey50;
  const grey100 = colors?.grey100;
  const grey200 = colors?.grey200;
  const grey300 = colors?.grey300;
  const grey500 = colors?.darkTextSecondary;
  const grey600 = colors?.heading;
  const grey700 = colors?.darkTextPrimary;
  const grey900 = colors?.textDark;
  const darkTextPrimary = colors?.darkTextPrimary;
  const darkLevel1 = colors?.darkLevel1;
  const darkLevel2 = colors?.darkLevel2;
  const darkBackground = colors?.darkBackground;
  const darkPaper = colors?.darkPaper;
  const darkTextSecondary = colors?.darkTextSecondary;
  const textDark = colors?.textDark;
  const paper = colors?.paper;
  const backgroundDefault = colors?.backgroundDefault;

  return {
    mode: navType,
    common: {
      black,
    },
    primary: {
      light: primaryLight,
      main: primaryMain,
      dark: primaryDark,
      200: primary200,
      800: primary800,
    },
    secondary: {
      light: secondaryLight,
      main: secondaryMain,
      dark: secondaryDark,
      200: secondary200,
      800: secondary800,
    },
    error: {
      light: errorLight,
      main: errorMain,
      dark: errorDark,
    },
    orange: {
      light: orangeLight,
      main: orangeMain,
      dark: orangeDark,
    },
    warning: {
      light: warningLight,
      main: warningMain,
      dark: warningDark,
    },
    success: {
      light: successLight,
      200: success200,
      main: successMain,
      dark: successDark,
    },
    grey: {
      50: grey50,
      100: grey100,
      200: grey200,
      300: grey300,
      500: grey500,
      600: grey600,
      700: grey700,
      900: grey900,
    },
    dark: {
      light: darkTextPrimary,
      main: darkLevel1,
      dark: darkLevel2,
      800: darkBackground,
      900: darkPaper,
    },
    text: {
      primary: darkTextPrimary,
      secondary: darkTextSecondary,
      dark: textDark,
      hint: grey100,
    },
    background: {
      paper,
      default: backgroundDefault,
    },
    card: {
      main: isDarkMode ? colors?.darkPrimaryMain : paper,
      light: isDarkMode ? colors?.darkPrimary200 : paper,
      hover: isDarkMode ? colors?.darkPrimary800 : paper,
    },
    asyncSelect: {
      main: isDarkMode ? colors?.darkPrimary800 : grey50,
    },
    canvasHeader: {
      deployLight: colors?.primaryLight,
      deployDark: colors?.primaryDark,
      saveLight: colors?.secondaryLight,
      saveDark: colors?.secondaryDark,
      settingsLight: colors?.grey300,
      settingsDark: colors?.grey700,
    },
    codeEditor: {
      main: isDarkMode ? colors?.darkPrimary800 : colors?.primaryLight,
    },
  };
}