/**
 * Colors used in theme
 * @param {JsonObject} theme theme customization object
 */

const themeColors = (theme) => ({
  textColorPrimary: theme.heading,
  textColorSecondary: theme.textDark,
  textColorHint: theme.darkTextSecondary,
  primaryColor: theme.primary,
  backgroundColor: theme.background,
  greyColor: theme.grey500
})

/**
 * Typography used in theme
 * @param {JsonObject} theme theme customization object
 */

const themeTypography = (theme) => {
  const { textColorPrimary, textColorSecondary, textColorHint } = themeColors(theme)
  
  return {
    fontFamily: theme?.customization?.fontFamily,
    h1: {
      fontSize: '2.125rem',
      fontWeight: 700,
      color: textColorPrimary
    },
    h2: {
      fontSize: '1.5rem',
      fontWeight: 700,
      color: textColorPrimary
    },
    h3: {
      fontSize: '1.25rem',
      fontWeight: 600,
      color: textColorPrimary
    },
    h4: {
      fontSize: '1rem',
      fontWeight: 600,
      color: textColorPrimary
    },
    h5: {
      fontSize: '0.875rem',
      fontWeight: 500,
      color: textColorPrimary
    },
    h6: {
      fontSize: '0.75rem',
      fontWeight: 500,
      color: textColorPrimary
    },
    subtitle1: {
      fontSize: '0.875rem',
      fontWeight: 500,
      color: textColorSecondary
    },
    subtitle2: {
      fontSize: '0.75rem',
      fontWeight: 400,
      color: textColorHint
    },
    body1: {
      fontSize: '1rem',
      fontWeight: 400,
      lineHeight: 1.5,
      color: textColorSecondary
    },
    body2: {
      fontSize: '0.875rem',
      letterSpacing: '0.00938em',
      fontWeight: 400,
      lineHeight: 1.43,
      color: textColorPrimary
    },
    button: {
      textTransform: 'none',
      fontWeight: 600
    },
    caption: {
      fontSize: '0.75rem',
      fontWeight: 400,
      lineHeight: 1.66,
      color: textColorHint
    }
  }
}

/**
 * Styles used in theme
 * @param {JsonObject} theme theme customization object
 */

export default function themeStyles(theme) {
  const { primaryColor, backgroundColor, greyColor } = themeColors(theme)
  const typography = themeTypography(theme)
  
  return {
    customInput: {
      marginTop: 1,
      marginBottom: 1,
      '& > label': {
        top: 23,
        left: 0,
        color: greyColor,
        '&[data-shrink="false"]': {
          top: 5
        }
      },
      '& > div > input': {
        padding: '30.5px 14px 11.5px !important'
      },
      '& legend': {
        display: 'none'
      },
      '& fieldset': {
        top: 0
      }
    },
    mainContent: {
      backgroundColor,
      width: '100%',
      minHeight: 'calc(100vh - 75px)',
      flexGrow: 1,
      padding: '20px',
      marginTop: '75px',
      marginRight: '20px',
      borderRadius: `${theme?.customization?.borderRadius}px`
    },
    menuCaption: {
      fontSize: '0.875rem',
      fontWeight: 500,
      color: primaryColor,
      padding: '6px',
      textTransform: 'capitalize',
      marginTop: '10px'
    },
    subMenuCaption: {
      fontSize: '0.6875rem',
      fontWeight: 500,
      color: greyColor,
      textTransform: 'capitalize'
    },
    commonAvatar: {
      cursor: 'pointer',
      borderRadius: '8px'
    },
    smallAvatar: {
      width: '22px',
      height: '22px',
      fontSize: '1rem'
    },
    mediumAvatar: {
      width: '34px',
      height: '34px',
      fontSize: '1.2rem'
    },
    largeAvatar: {
      width: '44px',
      height: '44px',
      fontSize: '1.5rem'
    },
    ...typography
  }
}

