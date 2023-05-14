import React, { forwardRef } from 'react'
import PropTypes, { InferProps } from 'prop-types'
import { Card, CardContent, CardHeader, Divider, Typography } from '@mui/material'

interface MainCardProps {
  border?: boolean
  boxShadow?: boolean
  children?: React.ReactNode
  content?: boolean
  contentClass?: string
  contentSX?: Record<string, unknown>
  darkTitle?: boolean
  secondary?: React.ReactNode | string | Record<string, unknown>
  shadow?: string
  sx?: Record<string, unknown>
  title?: React.ReactNode | string | Record<string, unknown>
}

function Header({ darkTitle, title, secondary }: InferProps<typeof Header.propTypes>) {
  return (
    <CardHeader
      title={
        darkTitle ? (
          <Typography variant="h3">{title}</Typography>
        ) : (
          <Typography variant="h5">{title}</Typography>
        )
      }
      action={secondary}
    />
  )
}

Header.propTypes = {
  darkTitle: PropTypes.bool,
  title: PropTypes.oneOfType([PropTypes.node, PropTypes.string, PropTypes.object]),
  secondary: PropTypes.oneOfType([PropTypes.node, PropTypes.string, PropTypes.object]),
}

function Content({ children, contentSX, contentClass }: InferProps<typeof Content.propTypes>) {
  return (
    <CardContent sx={contentSX} className={contentClass}>
      {children}
    </CardContent>
  )
}

Content.propTypes = {
  children: PropTypes.node,
  contentClass: PropTypes.string,
  contentSX: PropTypes.object,
}

const MainCard = forwardRef<HTMLDivElement, MainCardProps>(function MainCard(
  {
    border = true,
    boxShadow,
    children,
    content = true,
    contentClass,
    contentSX,
    darkTitle,
    secondary,
    shadow,
    sx,
    title,
    ...others
  },
  ref
) {
  const borderColor = border ? 'primary.275' : 'transparent'
  const boxShadowValue = boxShadow ? shadow || '0 2px 14px 0 rgb(32 40 45 / 8%)' : undefined

  return (
    <Card
      ref={ref}
      sx={{
        border: `1px solid`,
        borderColor,
        ':hover': {
          boxShadow: boxShadowValue,
        },
        ...sx,
      }}
      {...others}
    >
      {title && <Header darkTitle={darkTitle} title={title} secondary={secondary} />}
      {title && <Divider />}
      {content && <Content contentClass={contentClass} contentSX={contentSX}>{children}</Content>}
      {!content && children}
    </Card>
  )
})

MainCard.propTypes = {
  border: PropTypes.bool,
  boxShadow: PropTypes.bool,
  children: PropTypes.node,
  content: PropTypes.bool,
  contentClass: PropTypes.string,
  contentSX: PropTypes.object,
  darkTitle: PropTypes.bool,
  secondary: PropTypes.oneOfType([PropTypes.node, PropTypes.string, PropTypes.object]),
  shadow: PropTypes.string,
  sx: PropTypes.object,
  title: PropTypes.oneOfType([PropTypes.node, PropTypes.string, PropTypes.object]),
}

export default MainCard
