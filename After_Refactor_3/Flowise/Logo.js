import React from 'react'
import { useSelector } from 'react-redux'
import PropTypes from 'prop-types'
import styled from 'styled-components'

import { layout } from 'theme/sizes'
import { getCustomization } from 'store/selectors'
import { MEDIA_QUERY } from 'constants/styles'

import { Logo, LogoDark } from 'assets/images'

const LogoImg = styled.img`
  object-fit: contain;
  height: auto;
  width: ${layout.header.logo.width};
  ${MEDIA_QUERY.desktop`
    width: ${layout.header.logo.desktopWidth};
  `}
`

const LogoWrapper = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
`

const FlowiseLogo = ({ className }) => {
  const customization = useSelector(getCustomization)
  return (
    <LogoWrapper className={className}>
      <LogoImg
        src={customization.isDarkMode ? LogoDark : Logo}
        alt='Flowise Logo'
      />
    </LogoWrapper>
  )
}

FlowiseLogo.propTypes = {
  className: PropTypes.string,
}

FlowiseLogo.defaultProps = {
  className: '',
}

export default FlowiseLogo