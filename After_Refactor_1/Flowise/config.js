function getConfig(options = {}) {
  const {
    // The base URL of the application. Do not add a trailing slash.
    // Use an empty string for the root URL.
    baseUrl = '',
    // The default path to load when the application is first opened.
    defaultPath = '/chatflows',
    // The font family to use for all text on the page.
    fontFamily = "'Roboto', sans-serif",
    // The border radius to use for all components with rounded corners.
    borderRadius = 12,
  } = options;

  return {
    baseUrl,
    defaultPath,
    fontFamily,
    borderRadius,
  };
}

export default getConfig;