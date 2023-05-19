import PropTypes from 'prop-types';
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Box, Card, Divider, Grid, Typography } from '@mui/material';
import MuiBreadcrumbs from '@mui/material/Breadcrumbs';
import { useTheme } from '@mui/material/styles';
import { IconTallymark1 } from '@tabler/icons';
import AccountTreeTwoToneIcon from '@mui/icons-material/AccountTreeTwoTone';
import HomeIcon from '@mui/icons-material/Home';
import HomeTwoToneIcon from '@mui/icons-material/HomeTwoTone';

import config from 'config';
import { gridSpacing } from 'store/constant';

const linkStyles = {
  display: 'flex',
  color: 'grey.900',
  textDecoration: 'none',
  alignContent: 'center',
  alignItems: 'center'
};

const defaultProps = {
  card: true,
  divider: true,
  icon: false,
  icons: true,
  rightAlign: true,
  separator: IconTallymark1,
  title: true,
  titleBottom: false
};

const propTypes = {
  card: PropTypes.bool,
  divider: PropTypes.bool,
  icon: PropTypes.bool,
  icons: PropTypes.bool,
  maxItems: PropTypes.number,
  navigation: PropTypes.object,
  rightAlign: PropTypes.bool,
  separator: PropTypes.oneOfType([PropTypes.func, PropTypes.object]),
  title: PropTypes.bool,
  titleBottom: PropTypes.bool
};

const Breadcrumbs = (props) => {
  const theme = useTheme();
  const { card, divider, icon, icons, maxItems, navigation, rightAlign, separator, title, titleBottom } = {
    ...defaultProps,
    ...props
  };

  const iconStyle = {
    marginRight: theme.spacing(0.75),
    marginTop: `-${theme.spacing(0.25)}`,
    width: '1rem',
    height: '1rem',
    color: theme.palette.secondary.main
  };

  const [mainItem, setMainItem] = useState();
  const [collapseItem, setCollapseItem] = useState();

  const handleCollapseSelectedItem = (menu) => {
    if (menu.children) {
      menu.children.filter((collapse) => {
        if (collapse.type && collapse.type === 'collapse') {
          handleCollapseSelectedItem(collapse);
        } else if (collapse.type && collapse.type === 'item') {
          if (document.location.pathname === config.basename + collapse.url) {
            setMainItem(menu);
            setCollapseItem(collapse);
          }
        }
        return false;
      });
    }
  };

  useEffect(() => {
    navigation?.items?.forEach((menu) => {
      if (menu.type && menu.type === 'group') {
        handleCollapseSelectedItem(menu);
      }
      return false;
    });
  });

  const SeparatorIcon = separator;
  const separatorIcon = separator ? <SeparatorIcon stroke={1.5} size='1rem' /> : <IconTallymark1 stroke={1.5} size='1rem' />;

  let mainContent;
  let collapseContent;
  let breadcrumbContent = <Typography />;
  let collapseTitle = '';
  let CollapseIcon;
  let ItemIcon;

  if (mainItem && mainItem.type === 'collapse') {
    CollapseIcon = mainItem.icon ? mainItem.icon : AccountTreeTwoToneIcon;
    mainContent = (
      <Typography component={Link} to='#' variant='subtitle1' sx={linkStyles}>
        {icons && <CollapseIcon style={iconStyle} />}
        {mainItem.title}
      </Typography>
    );
  }

  if (collapseItem && collapseItem.type === 'item') {
    collapseTitle = collapseItem.title;

    ItemIcon = collapseItem.icon ? collapseItem.icon : AccountTreeTwoToneIcon;
    collapseContent = (
      <Typography variant='subtitle1' sx={{ ...linkStyles, color: 'grey.500' }}>
        {icons && <ItemIcon style={iconStyle} />}
        {collapseTitle}
      </Typography>
    );

    if (collapseItem.breadcrumbs !== false) {
      breadcrumbContent = (
        <Card
          sx={{
            border: 'none'
          }}
          {...props}
        >
          <Box sx={{ p: 2, pl: card === false ? 0 : 2 }}>
            <Grid container direction={rightAlign ? 'row' : 'column'} justifyContent={rightAlign ? 'space-between' : 'flex-start'} alignItems={rightAlign ? 'center' : 'flex-start'} spacing={1}>
              {title && !titleBottom && (
                <Grid item>
                  <Typography variant='h3' sx={{ fontWeight: 500 }}>
                    {collapseItem.title}
                  </Typography>
                </Grid>
              )}
              <Grid item>
                <MuiBreadcrumbs sx={{ '& .MuiBreadcrumbs-separator': { width: 16, ml: 1.25, mr: 1.25 } }} aria-label='breadcrumb' maxItems={maxItems || 8} separator={separatorIcon}>
                  <Typography component={Link} to='/' color='inherit' variant='subtitle1' sx={linkStyles}>
                    {icons && <HomeTwoToneIcon sx={iconStyle} />}
                    {icon && <HomeIcon sx={{ ...iconStyle, mr: 0 }} />}
                    {!icon && 'Dashboard'}
                  </Typography>
                  {mainContent}
                  {collapseContent}
                </MuiBreadcrumbs>
              </Grid>
              {title && titleBottom && (
                <Grid item>
                  <Typography variant='h3' sx={{ fontWeight: 500 }}>
                    {collapseItem.title}
                  </Typography>
                </Grid>
              )}
            </Grid>
          </Box>
          {card === false && divider !== false && <Divider sx={{ borderColor: theme.palette.primary.main, mb: gridSpacing }} />}
        </Card>
      );
    }
  }

  return breadcrumbContent;
};

Breadcrumbs.propTypes = propTypes;
Breadcrumbs.defaultProps = defaultProps;

export default Breadcrumbs;

