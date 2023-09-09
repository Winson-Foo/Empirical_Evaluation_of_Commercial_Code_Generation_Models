import config from 'config';

export const getActiveItem = (menuItems) => {
  const getMenuItems = (menu) => {
    if (menu.children) {
      menu.children.filter((item) => {
        if (item.type && item.type === 'collapse') {
          getMenuItems(item);
        } else if (item.type && item.type === 'item') {
          if (document.location.pathname === config.basename + item.url) {
            return item;
          }
        }
        return false;
      });
    }
  };

  const activeMenuItems = menuItems.map((menu) => {
    if (menu.type && menu.type === 'group') {
      return getMenuItems(menu);
    }
    return false;
  });
  return activeMenuItems.filter((x) => x)[0];
};

export const renderBreadcrumbsContent = ({
  item,
  main,
  icons,
  iconStyle,
  SeparatorIcon,
  itemTitle,
}) => {
  let mainContent;
  let itemContent;
  let breadcrumbContent = <Typography />;
  let CollapseIcon;
  let ItemIcon;

  // collapse item
  if (main && main.type === 'collapse') {
    CollapseIcon = main.icon ? main.icon : AccountTreeTwoToneIcon;
    mainContent = (
      <Typography component={Link} to="#" variant="subtitle1" sx={linkSX}>
        {icons && <CollapseIcon style={iconStyle} />}
        {main.title}
      </Typography>
    );
  }

  // items
  if (item && item.type === 'item') {
    itemTitle = item.title;

    ItemIcon = item.icon ? item.icon : AccountTreeTwoToneIcon;
    itemContent = (
      <Typography
        variant="subtitle1"
        sx={{
          display: 'flex',
          textDecoration: 'none',
          alignContent: 'center',
          alignItems: 'center',
          color: 'grey.500',
        }}
      >
        {icons && <ItemIcon style={iconStyle} />}
        {itemTitle}
      </Typography>
    );

    // main
    if (item.breadcrumbs !== false) {
      breadcrumbContent = (
        <Card
          sx={{
            border: 'none',
          }}
          {...others}
        >
          <Box sx={{ p: 2, pl: card === false ? 0 : 2 }}>
            <Grid
              container
              direction={rightAlign ? 'row' : 'column'}
              justifyContent={rightAlign ? 'space-between' : 'flex-start'}
              alignItems={rightAlign ? 'center' : 'flex-start'}
              spacing={1}
            >
              {title && !titleBottom && (
                <Grid item>
                  <Typography variant="h3" sx={{ fontWeight: 500 }}>
                    {item.title}
                  </Typography>
                </Grid>
              )}
              <Grid item>
                <MuiBreadcrumbs
                  sx={{ '& .MuiBreadcrumbs-separator': { width: 16, ml: 1.25, mr: 1.25 } }}
                  aria-label="breadcrumb"
                  maxItems={maxItems || 8}
                  separator={separatorIcon}
                >
                  <Typography
                    component={Link}
                    to="/"
                    color="inherit"
                    variant="subtitle1"
                    sx={linkSX}
                  >
                    {icons && <HomeTwoToneIcon sx={iconStyle} />}
                    {icon && <HomeIcon sx={{ ...iconStyle, mr: 0 }} />}
                    {!icon && 'Dashboard'}
                  </Typography>
                  {mainContent}
                  {itemContent}
                </MuiBreadcrumbs>
              </Grid>
              {title && titleBottom && (
                <Grid item>
                  <Typography variant="h3" sx={{ fontWeight: 500 }}>
                    {item.title}
                  </Typography>
                </Grid>
              )}
            </Grid>
          </Box>
          {card === false && divider !== false && (
            <Divider sx={{ borderColor: theme.palette.primary.main, mb: gridSpacing }} />
          )}
        </Card>
      );
    }
  }

  return breadcrumbContent;
};
```

NestedBreadcrumbs.js:

```
import PropTypes from 'prop-types';
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';

// material-ui
import { useTheme } from '@mui/material/styles';
import { Box, Card, Divider, Grid, Typography } from '@mui/material';
import MuiBreadcrumbs from '@mui/material/Breadcrumbs';

// project imports
import { gridSpacing } from 'store/constant';
import { getActiveItem, renderBreadcrumbsContent } from './BreadcrumbUtils';

// assets
import { IconTallymark1 } from '@tabler/icons';
import AccountTreeTwoToneIcon from '@mui/icons-material/AccountTreeTwoTone';
import HomeIcon from '@mui/icons-material/Home';
import HomeTwoToneIcon from '@mui/icons-material/HomeTwoTone';

const linkSX = {
  display: 'flex',
  color: 'grey.900',
  textDecoration: 'none',
  alignContent: 'center',
  alignItems: 'center',
};

// ==============================|| NESTED BREADCRUMBS ||============================== //

const NestedBreadcrumbs = ({
  card,
  divider,
  icon,
  icons,
  maxItems,
  menuItems,
  rightAlign,
  separator,
  title,
  titleBottom,
  ...others
}) => {
  const theme = useTheme();

  const iconStyle = {
    marginRight: theme.spacing(0.75),
    marginTop: `-${theme.spacing(0.25)}`,
    width: '1rem',
    height: '1rem',
    color: theme.palette.secondary.main,
  };

  const [main, setMain] = useState();
  const [item, setItem] = useState();

  useEffect(() => {
    const activeItem = getActiveItem(menuItems);
    setMain(activeItem.main);
    setItem(activeItem.item);
  }, [menuItems]);

  // item separator
  const SeparatorIcon = separator;
  const separatorIcon = separator ? <SeparatorIcon stroke={1.5} size="1rem" /> : <IconTallymark1 stroke={1.5} size="1rem" />;

  return renderBreadcrumbsContent({
    item,
    main,
    icons,
    iconStyle,
    SeparatorIcon,
    itemTitle,
    card,
    divider,
    maxItems,
    rightAlign,
    separator,
    title,
    titleBottom,
    ...others,
  });
};

NestedBreadcrumbs.propTypes = {
  card: PropTypes.bool,
  divider: PropTypes.bool,
  icon: PropTypes.bool,
  icons: PropTypes.bool,
  maxItems: PropTypes.number,
  menuItems: PropTypes.array,
  rightAlign: PropTypes.bool,
  separator: PropTypes.oneOfType([PropTypes.func, PropTypes.object]),
  title: PropTypes.bool,
  titleBottom: PropTypes.bool,
};

export default NestedBreadcrumbs;