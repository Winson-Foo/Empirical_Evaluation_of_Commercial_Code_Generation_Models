import PropTypes from 'prop-types';
import { forwardRef } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Divider,
  Typography,
} from '@mui/material';
import { useTheme } from '@mui/material/styles';

const headerStyles = {
  '& .MuiCardHeader-action': {
    marginRight: 0,
  },
};

const MainCard = forwardRef(
  (
    {
      border,
      boxShadow,
      children,
      content,
      contentClass,
      contentSX,
      darkTitle,
      secondary,
      shadow,
      sx,
      title,
      ...props
    },
    ref,
  ) => {
    const theme = useTheme();
    const borderColor = `${theme.palette.primary[200]}75`;
    const boxShadowColor = shadow || '0 2px 14px 0 rgb(32 40 45 / 8%)';

    return (
      <Card
        ref={ref}
        {...props}
        sx={{
          border: border ? '1px solid' : 'none',
          borderColor,
          ':hover': {
            boxShadow: boxShadow ? boxShadowColor : 'inherit',
          },
          ...sx,
        }}
      >
        {/* Card Header */}
        {title && (
          <>
            <CardHeader
              sx={headerStyles}
              title={
                darkTitle ? (
                  <Typography variant='h3'>{title}</Typography>
                ) : (
                  title
                )
              }
              action={secondary}
            />
            <Divider />
          </>
        )}

        {/* Card Content */}
        {content ? (
          <CardContent sx={contentSX} className={contentClass}>
            {children}
          </CardContent>
        ) : (
          children
        )}
      </Card>
    );
  },
);

MainCard.propTypes = {
  border: PropTypes.bool,
  boxShadow: PropTypes.bool,
  children: PropTypes.node,
  content: PropTypes.bool,
  contentClass: PropTypes.string,
  contentSX: PropTypes.object,
  darkTitle: PropTypes.bool,
  secondary: PropTypes.oneOfType([
    PropTypes.node,
    PropTypes.string,
    PropTypes.object,
  ]),
  shadow: PropTypes.string,
  sx: PropTypes.object,
  title: PropTypes.oneOfType([
    PropTypes.node,
    PropTypes.string,
    PropTypes.object,
  ]),
};

MainCard.defaultProps = {
  border: true,
  boxShadow: true,
  content: true,
};

export default MainCard;