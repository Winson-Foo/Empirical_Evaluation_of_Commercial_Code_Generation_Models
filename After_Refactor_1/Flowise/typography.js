// constants.js
export const FONT_WEIGHT = {
  thin: 100,
  light: 300,
  regular: 400,
  medium: 500,
  bold: 700,
  black: 900,
};

export const LINE_HEIGHT = {
  single: 1,
  narrow: 1.25,
  standard: 1.5,
  wide: 2,
};

// body.js
export default function body(theme) {
  return {
    fontSize: '0.875rem',
    fontWeight: FONT_WEIGHT.regular,
    lineHeight: LINE_HEIGHT.standard,
  };
}

// caption.js
export default function caption(theme) {
  return {
    fontSize: '0.75rem',
    fontWeight: FONT_WEIGHT.regular,
    color: theme.darkTextSecondary,
  };
}

// heading.js
export default function heading(theme, fontSize, fontWeight) {
  return {
    fontSize,
    color: theme.heading,
    fontWeight,
  };
}

// input.js
export default function input(theme) {
  return {
    marginTop: 1,
    marginBottom: 1,
    '& > label': {
      top: 23,
      left: 0,
      color: theme.grey500,
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
  };
}

// menu.js
export default function menu(theme) {
  return {
    fontSize: '0.875rem',
    fontWeight: FONT_WEIGHT.medium,
    color: theme.heading,
    padding: '6px',
    textTransform: 'capitalize',
    marginTop: '10px',
  };
}

// subtitle.js
export default function subtitle(theme, fontSize, fontWeight) {
  return {
    fontSize,
    fontWeight,
    color: theme.textDark,
  };
}

// typography.js
import body from './body';
import caption from './caption'
import heading from './heading';
import input from './input';
import menu from './menu';
import subtitle from './subtitle';

export default function themeTypography(theme) {
  return {
    fontFamily: theme?.customization?.fontFamily,
    h6: heading(theme, '0.75rem', FONT_WEIGHT.medium),
    h5: heading(theme, '0.875rem', FONT_WEIGHT.medium),
    h4: heading(theme, '1rem', FONT_WEIGHT.bold),
    h3: heading(theme, '1.25rem', FONT_WEIGHT.bold),
    h2: heading(theme, '1.5rem', FONT_WEIGHT.bold),
    h1: heading(theme, '2.125rem', FONT_WEIGHT.bold),
    subtitle1: subtitle(theme, '0.875rem', FONT_WEIGHT.medium),
    subtitle2: subtitle(theme, '0.75rem', FONT_WEIGHT.regular),
    caption: caption(theme),
    body1: body(theme),
    body2: {
      letterSpacing: '0em',
      fontWeight: FONT_WEIGHT.regular,
      lineHeight: LINE_HEIGHT.standard,
      color: theme.darkTextPrimary,
    },
    button: {
      textTransform: 'capitalize',
    },
    customInput: input(theme),
    mainContent: {
      backgroundColor: theme.background,
      width: '100%',
      minHeight: 'calc(100vh - 75px)',
      flexGrow: 1,
      padding: '20px',
      marginTop: '75px',
      marginRight: '20px',
      borderRadius: `${theme?.customization?.borderRadius}px`,
    },
    menuCaption: menu(theme),
    subMenuCaption: {
      fontSize: '0.6875rem',
      fontWeight: FONT_WEIGHT.medium,
      color: theme.darkTextSecondary,
      textTransform: 'capitalize',
    },
    commonAvatar: {
      cursor: 'pointer',
      borderRadius: '8px',
    },
    smallAvatar: {
      width: '22px',
      height: '22px',
      fontSize: '1rem',
    },
    mediumAvatar: {
      width: '34px',
      height: '34px',
      fontSize: '1.2rem',
    },
    largeAvatar: {
      width: '44px',
      height: '44px',
      fontSize: '1.5rem',
    },
  };
}

