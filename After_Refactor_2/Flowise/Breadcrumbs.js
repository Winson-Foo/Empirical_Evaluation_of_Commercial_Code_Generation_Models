import PropTypes from 'prop-types'
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { useTheme } from '@mui/material/styles'
import { Box, Card, Divider, Grid, Typography } from '@mui/material'
import MuiBreadcrumbs from '@mui/material/Breadcrumbs'
import { IconTallymark1 } from '@tabler/icons'
import AccountTreeTwoToneIcon from '@mui/icons-material/AccountTreeTwoTone'
import HomeIcon from '@mui/icons-material/Home'
import HomeTwoToneIcon from '@mui/icons-material/HomeTwoTone'

const linkStyles = {
  display: 'flex',
  color: 'grey.900',
  textDecoration: 'none',
  alignContent: 'center',
  alignItems: 'center'
}

const BreadcrumbsLink = ({ to, title, icon: Icon, withIcon }) => {
  const theme = useTheme()

  const iconStyle = {
    marginRight: theme.spacing(0.75),
    marginTop: `-${theme.spacing(0.25)}`,
    width: '1rem',
    height: '1rem',
    color: theme.palette.secondary.main
  }

  return (
    <Typography component={Link} to={to} variant='subtitle1' sx={linkStyles}>
      {withIcon && <Icon style={iconStyle} />}
      {title}
    </Typography>
  )
}

BreadcrumbsLink.propTypes = {
  to: PropTypes.string.isRequired,
  title: PropTypes.string.isRequired,
  icon: PropTypes.elementType,
  withIcon: PropTypes.bool.isRequired
}

const Breadcrumbs = ({
  card = true,
  divider = true,
  icon = false,
  icons = true,
  maxItems = 8,
  navigation,
  rightAlign = false,
  separator: SeparatorIcon = IconTallymark1,
  title = true,
  titleBottom = false,
  ...others
}) => {
  const [mainCollapse, setMainCollapse] = useState()
  const [activeItem, setActiveItem] = useState()

  useEffect(() => {
    const selectActiveItem = (menu) => {
      if (menu.children) {
        menu.children.filter((child) => {
          if (child.type === 'collapse') {
            selectActiveItem(child)
          } else if (child.type === 'item') {
            if (document.location.pathname === config.basename + child.url) {
              setMainCollapse(menu)
              setActiveItem(child)
            }
          }
          return false
        })
      }
    }

    navigation?.items?.forEach((menu) => {
      if (menu.type === 'group') {
        selectActiveItem(menu)
      }
      return false
    })
  }, [navigation])

  const theme = useTheme()

  const Separator = SeparatorIcon
  const separatorIcon = <Separator stroke={1.5} size='1rem' />

  const mainContent = mainCollapse?.type === 'collapse' && (
    <BreadcrumbsLink
      to='#'
      title={mainCollapse.title}
      icon={mainCollapse.icon || AccountTreeTwoToneIcon}
      withIcon={icons}
    />
  )

  const itemContent = activeItem?.type === 'item' && (
    <Typography
      variant='subtitle1'
      sx={{
        display: 'flex',
        textDecoration: 'none',
        alignContent: 'center',
        alignItems: 'center',
        color: 'grey.500'
      }}
    >
      {icons && <activeItem.icon style={{ ...iconStyle }} />}
      {activeItem.title}
    </Typography>
  )

  const titleContent = title && (
    <Grid item>
      <Typography variant='h3' sx={{ fontWeight: 500 }}>
        {activeItem?.title || ''}
      </Typography>
    </Grid>
  )

  const breadcrumbContent = (
    <Card
      sx={{
        border: 'none'
      }}
      {...others}
    >
      <Box sx={{ p: 2, pl: card ? 2 : 0 }}>
        <Grid
          container
          direction={rightAlign ? 'row' : 'column'}
          justifyContent={rightAlign ? 'space-between' : 'flex-start'}
          alignItems={rightAlign ? 'center' : 'flex-start'}
          spacing={1}
        >
          <Grid item>
            <MuiBreadcrumbs
              sx={{ '& .MuiBreadcrumbs-separator': { width: 16, ml: 1.25, mr: 1.25 } }}
              aria-label='breadcrumb'
              maxItems={maxItems}
              separator={separatorIcon}
            >
              <BreadcrumbsLink
                to='/'
                title='Dashboard'
                icon={icon ? HomeIcon : HomeTwoToneIcon}
                withIcon={icons}
              />
              {mainContent}
              {itemContent}
            </MuiBreadcrumbs>
          </Grid>
          {titleBottom && titleContent}
        </Grid>
      </Box>
      {!card && divider && (
        <Divider sx={{ borderColor: theme.palette.primary.main, mb: gridSpacing }} />
      )}
    </Card>
  )

  return breadcrumbContent
}

Breadcrumbs.propTypes = {
  card: PropTypes.bool,
  divider: PropTypes.bool,
  icon: PropTypes.bool,
  icons: PropTypes.bool,
  maxItems: PropTypes.number,
  navigation: PropTypes.object,
  rightAlign: PropTypes.bool,
  separator: PropTypes.elementType,
  title: PropTypes.bool,
  titleBottom: PropTypes.bool
}

export default Breadcrumbs

