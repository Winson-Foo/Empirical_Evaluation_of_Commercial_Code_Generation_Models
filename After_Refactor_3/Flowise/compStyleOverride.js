const buttonStyleOverrides = () => ({
  root: {
    fontWeight: 500,
    borderRadius: '4px'
  }
});

const svgIconStyleOverrides = (theme) => ({
  root: {
    color: theme?.customization?.isDarkMode ? theme.colors?.paper : 'inherit',
    background: theme?.customization?.isDarkMode ? theme.colors?.darkPrimaryLight : 'inherit'
  }
});

const paperDefaultProps = () => ({
  elevation: 0
});

const paperStyleOverrides = (theme) => ({
  root: {
    backgroundImage: 'none'
  },
  rounded: {
    borderRadius: `${theme?.customization?.borderRadius}px`
  }
});

const cardHeaderStyleOverrides = (theme) => ({
  root: {
    color: theme.colors?.textDark,
    padding: '24px'
  },
  title: {
    fontSize: '1.125rem'
  }
});

const cardContentStyleOverrides = () => ({
  root: {
    padding: '24px'
  }
});

const cardActionsStyleOverrides = () => ({
  root: {
    padding: '24px'
  }
});

const listItemButtonStyleOverrides = (theme) => ({
  root: {
    color: theme.darkTextPrimary,
    paddingTop: '10px',
    paddingBottom: '10px',
    '&.Mui-selected': {
      color: theme.menuSelected,
      backgroundColor: theme.menuSelectedBack,
      '&:hover': {
        backgroundColor: theme.menuSelectedBack
      },
      '& .MuiListItemIcon-root': {
        color: theme.menuSelected
      }
    },
    '&:hover': {
      backgroundColor: theme.menuSelectedBack,
      color: theme.menuSelected,
      '& .MuiListItemIcon-root': {
        color: theme.menuSelected
      }
    }
  }
});

const listItemIconStyleOverrides = (theme) => ({
  root: {
    color: theme.darkTextPrimary,
    minWidth: '36px'
  }
});

const listItemTextStyleOverrides = (theme) => ({
  primary: {
    color: theme.textDark
  }
});

const inputBaseStyleOverrides = (theme) => ({
  input: {
    color: theme.textDark,
    '&::placeholder': {
      color: theme.darkTextSecondary,
      fontSize: '0.875rem'
    }
  }
});

const outlinedInputStyleOverrides = (theme) => ({
  root: {
    background: theme?.customization?.isDarkMode ? theme.colors?.darkPrimary800 : bgColor,
    borderRadius: `${theme?.customization?.borderRadius}px`,
    '& .MuiOutlinedInput-notchedOutline': {
      borderColor: theme.colors?.grey400
    },
    '&:hover $notchedOutline': {
      borderColor: theme.colors?.primaryLight
    },
    '&.MuiInputBase-multiline': {
      padding: 1
    }
  },
  input: {
    fontWeight: 500,
    background: theme?.customization?.isDarkMode ? theme.colors?.darkPrimary800 : bgColor,
    padding: '15.5px 14px',
    borderRadius: `${theme?.customization?.borderRadius}px`,
    '&.MuiInputBase-inputSizeSmall': {
      padding: '10px 14px',
      '&.MuiInputBase-inputAdornedStart': {
        paddingLeft: 0
      }
    }
  },
  inputAdornedStart: {
    paddingLeft: 4
  },
  notchedOutline: {
    borderRadius: `${theme?.customization?.borderRadius}px`
  }
});

const sliderStyleOverrides = (theme) => ({
  root: {
    '&.Mui-disabled': {
      color: theme.colors?.grey300
    }
  },
  mark: {
    backgroundColor: theme.paper,
    width: '4px'
  },
  valueLabel: {
    color: theme?.colors?.primaryLight
  }
});

const dividerStyleOverrides = (theme) => ({
  root: {
    borderColor: theme.divider,
    opacity: 1
  }
});

const avatarStyleOverrides = (theme) => ({
  root: {
    color: theme.colors?.primaryDark,
    background: theme.colors?.primary200
  }
});

const chipStyleOverrides = () => ({
  root: {
    '&.MuiChip-deletable .MuiChip-deleteIcon': {
      color: 'inherit'
    }
  }
});

const tooltipStyleOverrides = (theme) => ({
  tooltip: {
    color: theme?.customization?.isDarkMode ? theme.colors?.paper : theme.paper,
    background: theme.colors?.grey700
  }
});

const autocompleteStyleOverrides = (theme) => ({
  option: {
    '&:hover': {
      background: theme?.customization?.isDarkMode ? '#233345 !important' : ''
    }
  }
});

export default function componentStyleOverrides(theme) {
  const bgColor = theme.colors?.grey50;

  return {
    MuiButton: {
      styleOverrides: buttonStyleOverrides()
    },
    MuiSvgIcon: {
      styleOverrides: svgIconStyleOverrides(theme)
    },
    MuiPaper: {
      defaultProps: paperDefaultProps(),
      styleOverrides: paperStyleOverrides(theme)
    },
    MuiCardHeader: {
      styleOverrides: cardHeaderStyleOverrides(theme)
    },
    MuiCardContent: {
      styleOverrides: cardContentStyleOverrides()
    },
    MuiCardActions: {
      styleOverrides: cardActionsStyleOverrides()
    },
    MuiListItemButton: {
      styleOverrides: listItemButtonStyleOverrides(theme)
    },
    MuiListItemIcon: {
      styleOverrides: listItemIconStyleOverrides(theme)
    },
    MuiListItemText: {
      styleOverrides: listItemTextStyleOverrides(theme)
    },
    MuiInputBase: {
      styleOverrides: inputBaseStyleOverrides(theme)
    },
    MuiOutlinedInput: {
      styleOverrides: outlinedInputStyleOverrides(theme)
    },
    MuiSlider: {
      styleOverrides: sliderStyleOverrides(theme)
    },
    MuiDivider: {
      styleOverrides: dividerStyleOverrides(theme)
    },
    MuiAvatar: {
      styleOverrides: avatarStyleOverrides(theme)
    },
    MuiChip: {
      styleOverrides: chipStyleOverrides()
    },
    MuiTooltip: {
      styleOverrides: tooltipStyleOverrides(theme)
    },
    MuiAutocomplete: {
      styleOverrides: autocompleteStyleOverrides(theme)
    }
  };
}

