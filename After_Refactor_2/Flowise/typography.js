const typography = (theme) => {
  const fontFamily = theme?.customization?.fontFamily || 'inherit';
  const borderRadius = theme?.customization?.borderRadius || 0;
  const headingColor = theme?.heading || 'inherit';
  const darkTextPrimaryColor = theme?.darkTextPrimary || 'inherit';
  const darkTextSecondaryColor = theme?.darkTextSecondary || 'inherit';
  const grey500Color = theme?.grey500 || 'inherit';
  const backgroundColor = theme?.background || 'inherit';

  return {
    // Headings
    h1: {
      fontSize: '2.125rem',
      fontWeight: 700,
      color: headingColor,
      fontFamily,
    },
    h2: {
      fontSize: '1.5rem',
      fontWeight: 700,
      color: headingColor,
      fontFamily,
    },
    h3: {
      fontSize: '1.25rem',
      fontWeight: 600,
      color: headingColor,
      fontFamily,
    },
    h4: {
      fontSize: '1rem',
      fontWeight: 600,
      color: headingColor,
      fontFamily,
    },
    h5: {
      fontSize: '0.875rem',
      fontWeight: 500,
      color: headingColor,
      fontFamily,
    },
    h6: {
      fontSize: '0.75rem',
      fontWeight: 500,
      color: headingColor,
      fontFamily,
    },
    // Body Text
    body1: {
      fontSize: '0.875rem',
      fontWeight: 400,
      fontFamily,
      lineHeight: '1.334em',
    },
    body2: {
      fontSize: '0.875rem',
      fontWeight: 400,
      fontFamily,
      letterSpacing: '0em',
      lineHeight: '1.5em',
      color: darkTextPrimaryColor,
    },
    // Subtitles
    subtitle1: {
      fontSize: '0.875rem',
      fontWeight: 500,
      color: darkTextPrimaryColor,
      fontFamily,
    },
    subtitle2: {
      fontSize: '0.75rem',
      fontWeight: 400,
      color: darkTextSecondaryColor,
      fontFamily,
    },
    // Caption
    caption: {
      fontSize: '0.75rem',
      fontWeight: 400,
      color: darkTextSecondaryColor,
      fontFamily,
    },
    // Button
    button: {
      textTransform: 'capitalize',
    },
    // Custom Input
    customInput: {
      marginTop: 1,
      marginBottom: 1,
      '& > label': {
        top: 23,
        left: 0,
        color: grey500Color,
        '&[data-shrink="false"]': {
          top: 5,
        },
      },
      '& > div > input': {
        padding: '30.5px 14px 11.5px !important',
      },
      '& legend': {
        display: 'none',
      },
      '& fieldset': {
        top: 0,
      },
    },
    // Main Content
    mainContent: {
      backgroundColor,
      width: '100%',
      flexGrow: 1,
      padding: '20px',
      marginTop: '75px',
      marginRight: '20px',
      borderRadius: `${borderRadius}px`,
      minHeight: 'calc(100vh - 75px)',
    },
    // Menu Caption
    menuCaption: {
      fontSize: '0.875rem',
      fontWeight: 500,
      color: headingColor,
      padding: '6px',
      textTransform: 'capitalize',
      marginTop: '10px',
      fontFamily,
    },
    // SubMenu Caption
    subMenuCaption: {
      fontSize: '0.6875rem',
      fontWeight: 500,
      color: darkTextSecondaryColor,
      textTransform: 'capitalize',
      fontFamily,
    },
    // Common Avatar
    commonAvatar: {
      cursor: 'pointer',
      borderRadius: '8px',
    },
    // Small Avatar
    smallAvatar: {
      width: '22px',
      height: '22px',
      fontSize: '1rem',
    },
    // Medium Avatar
    mediumAvatar: {
      width: '34px',
      height: '34px',
      fontSize: '1.2rem',
    },
    // Large Avatar
    largeAvatar: {
      width: '44px',
      height: '44px',
      fontSize: '1.5rem',
    },
  };
};

export default typography;