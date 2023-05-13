function getMode(theme) {
  return theme?.customization?.navType;
}

function getCommon(theme) {
  return {
    black: theme.colors?.darkPaper || '#000000'
  }
}

function getPrimary(theme) {
  const colors = theme.colors || {};
  const isDarkMode = theme.customization.isDarkMode;

  return {
    light: isDarkMode ? colors.darkPrimaryLight || '#000000' : colors.primaryLight || '#ffffff',
    main: colors.primaryMain || '#000000',
    dark: isDarkMode ? colors.darkPrimaryDark || '#000000' : colors.primaryDark || '#ffffff',
    200: isDarkMode ? colors.darkPrimary200 || '#000000' : colors.primary200 || '#ffffff',
    800: isDarkMode ? colors.darkPrimary800 || '#000000' : colors.primary800 || '#ffffff',
  };
}

function getSecondary(theme) {
  const colors = theme.colors || {};
  const isDarkMode = theme.customization.isDarkMode;

  return {
    light: isDarkMode ? colors.darkSecondaryLight || '#000000' : colors.secondaryLight || '#ffffff',
    main: isDarkMode ? colors.darkSecondaryMain || '#000000' : colors.secondaryMain || '#ffffff',
    dark: isDarkMode ? colors.darkSecondaryDark || '#000000' : colors.secondaryDark || '#ffffff',
    200: colors.secondary200 || '#000000',
    800: colors.secondary800 || '#ffffff',
  };
}

function getError(theme) {
  const colors = theme.colors || {};

  return {
    light: colors.errorLight || '#000000',
    main: colors.errorMain || '#000000',
    dark: colors.errorDark || '#000000',
  };
}

function getOrange(theme) {
  const colors = theme.colors || {};

  return {
    light: colors.orangeLight || '#000000',
    main: colors.orangeMain || '#000000',
    dark: colors.orangeDark || '#000000',
  };
}

function getWarning(theme) {
  const colors = theme.colors || {};

  return {
    light: colors.warningLight || '#000000',
    main: colors.warningMain || '#000000',
    dark: colors.warningDark || '#000000',
  };
}

function getSuccess(theme) {
  const colors = theme.colors || {};
  const isDarkMode = theme.customization.isDarkMode;

  return {
    light: colors.successLight || '#000000',
    200: colors.success200 || '#000000',
    main: colors.successMain || '#000000',
    dark: isDarkMode ? colors.darkSuccessDark || '#000000' : colors.successDark || '#ffffff',
  };
}

function getGrey(theme) {
  const colors = theme.colors || {};

  return {
    50: colors.grey50 || '#000000',
    100: colors.grey100 || '#000000',
    200: colors.grey200 || '#000000',
    300: colors.grey300 || '#000000',
    500: theme.darkTextSecondary || '#000000',
    600: theme.heading || '#000000',
    700: theme.darkTextPrimary || '#000000',
    900: theme.textDark || '#000000',
  };
}

function getDark(theme) {
  const colors = theme.colors || {};

  return {
    light: colors.darkTextPrimary || '#000000',
    main: colors.darkLevel1 || '#000000',
    dark: colors.darkLevel2 || '#000000',
    800: colors.darkBackground || '#000000',
    900: colors.darkPaper || '#000000',
  };
}

function getText(theme) {
  const colors = theme.colors || {};

  return {
    primary: theme.darkTextPrimary || '#000000',
    secondary: theme.darkTextSecondary || '#000000',
    dark: theme.textDark || '#000000',
    hint: colors.grey100 || '#000000',
  };
}

function getBackground(theme) {
  const colors = theme.colors || {};

  return {
    paper: theme.paper || '#ffffff',
    default: theme.backgroundDefault || '#ffffff',
  };
}

function getCard(theme) {
  const colors = theme.colors || {};
  const isDarkMode = theme.customization.isDarkMode;

  return {
    main: isDarkMode ? colors.darkPrimaryMain || colors.paper || '#000000' : colors.paper || '#ffffff',
    light: isDarkMode ? colors.darkPrimary200 || colors.paper || '#000000' : colors.paper || '#ffffff',
    hover: isDarkMode ? colors.darkPrimary800 || colors.paper || '#000000' : colors.paper || '#ffffff',
  };
}

function getAsyncSelect(theme) {
  const colors = theme.colors || {};

  return {
    main: theme.customization.isDarkMode ? colors.darkPrimary800 || colors.grey50 || '#000000' : colors.grey50 || '#ffffff',
  };
}

function getCanvasHeader(theme) {
  const colors = theme.colors || {};

  return {
    deployLight: colors.primaryLight || '#000000',
    deployDark: colors.primaryDark || '#000000',
    saveLight: colors.secondaryLight || '#000000',
    saveDark: colors.secondaryDark || '#000000',
    settingsLight: colors.grey300 || '#000000',
    settingsDark: colors.grey700 || '#000000',
  };
}

function getCodeEditor(theme) {
  const colors = theme.colors || {};

  return {
    main: theme.customization.isDarkMode ? colors.darkPrimary800 || colors.primaryLight || '#000000' : colors.primaryLight || '#ffffff',
  };
}

export default function themePalette(theme) {
  return {
    mode: getMode(theme),
    common: getCommon(theme),
    primary: getPrimary(theme),
    secondary: getSecondary(theme),
    error: getError(theme),
    orange: getOrange(theme),
    warning: getWarning(theme),
    success: getSuccess(theme),
    grey: getGrey(theme),
    dark: getDark(theme),
    text: getText(theme),
    background: getBackground(theme),
    card: getCard(theme),
    asyncSelect: getAsyncSelect(theme),
    canvasHeader: getCanvasHeader(theme),
    codeEditor: getCodeEditor(theme),
  };
}