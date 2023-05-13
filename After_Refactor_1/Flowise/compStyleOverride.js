export default function componentStyleOverrides(theme) {
  const bgColor = theme.colors?.grey50;

  const buttonStyles = {
    root: {
      fontWeight: 500,
      borderRadius: '4px'
    }
  };

  const svgIconStyles = {
    root: {
      color: theme?.customization?.isDarkMode
        ? theme.colors?.paper
        : 'inherit',
      background: theme?.customization?.isDarkMode
        ? theme.colors?.darkPrimaryLight
        : 'inherit'
    }
  };

  const paperStyles = {
    defaultProps: {
      elevation: 0
    },
    styleOverrides: {
      root: {
        backgroundImage: 'none'
      },
      rounded: {
        borderRadius: `${theme?.customization?.borderRadius}px`
      }
    }
  };

  const cardHeaderStyles = {
    root: {
      color: theme.colors?.textDark,
      padding: '24px'
    },
    title: {
      fontSize: '1.125rem'
    }
  };

  const cardContentStyles = {
    root: {
      padding: '24px'
    }
  };

  const cardActionsStyles = {
    root: {
      padding: '24px'
    }
  };

  const listItemButtonStyles = {
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
  };

  const listItemIconStyles = {
    root: {
      color: theme.darkTextPrimary,
      minWidth: '36px'
    }
  };

  const listItemTextStyles = {
    primary: {
      color: theme.textDark
    }
  };

  const inputBaseStyles = {
    input: {
      color: theme.textDark,
      '&::placeholder': {
        color: theme.darkTextSecondary,
        fontSize: '0.875rem'
      }
    }
  };

  const outlinedInputStyles = {
    root: {
      background: theme?.customization?.isDarkMode
        ? theme.colors?.darkPrimary800
        : bgColor,
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
      background: theme?.customization?.isDarkMode
        ? theme.colors?.darkPrimary800
        : bgColor,
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
  };

  const sliderStyles = {
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
  };

  const dividerStyles = {
    root: {
      borderColor: theme.divider,
      opacity: 1
    }
  };

  const avatarStyles = {
    root: {
      color: theme.colors?.primaryDark,
      background: theme.colors?.primary200
    }
  };

  const chipStyles = {
    root: {
      '&.MuiChip-deletable .MuiChip-deleteIcon': {
        color: 'inherit'
      }
    }
  };

  const tooltipStyles = {
    tooltip: {
      color: theme?.customization?.isDarkMode
        ? theme.colors?.paper
        : theme.paper,
      background: theme.colors?.grey700
    }
  };

  const autocompleteStyles = {
    option: {
      '&:hover': {
        background: theme?.customization?.isDarkMode ? '#233345 !important' : ''
      }
    }
  };

  return {
    MuiButton: { styleOverrides: buttonStyles },
    MuiSvgIcon: { styleOverrides: svgIconStyles },
    MuiPaper: paperStyles,
    MuiCardHeader: { styleOverrides: cardHeaderStyles },
    MuiCardContent: { styleOverrides: cardContentStyles },
    MuiCardActions: { styleOverrides: cardActionsStyles },
    MuiListItemButton: { styleOverrides: listItemButtonStyles },
    MuiListItemIcon: { styleOverrides: listItemIconStyles },
    MuiListItemText: { styleOverrides: listItemTextStyles },
    MuiInputBase: { styleOverrides: inputBaseStyles },
    MuiOutlinedInput: { styleOverrides: outlinedInputStyles },
    MuiSlider: { styleOverrides: sliderStyles },
    MuiDivider: { styleOverrides: dividerStyles },
    MuiAvatar: { styleOverrides: avatarStyles },
    MuiChip: { styleOverrides: chipStyles },
    MuiTooltip: { styleOverrides: tooltipStyles },
    MuiAutocomplete: { styleOverrides: autocompleteStyles }
  };
}