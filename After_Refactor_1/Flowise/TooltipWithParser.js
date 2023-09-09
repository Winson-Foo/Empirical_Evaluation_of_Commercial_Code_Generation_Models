import React from 'react';
import PropTypes from 'prop-types';
import { useSelector } from 'react-redux';
import { Tooltip, IconButton } from '@mui/material';
import { Info } from '@mui/icons-material';
import parser from 'html-react-parser';

const styles = {
  iconButton: {
    height: 25,
    width: 25,
  },
};

const iconStyles = {
  background: 'transparent',
  height: 18,
  width: 18,
};

const InfoTooltip = ({ title }) => {
  const customization = useSelector((state) => state.customization);

  return (
    <Tooltip title={parser(title)} placement='right'>
      <IconButton sx={styles.iconButton}>
        <Info
          style={{
            ...iconStyles,
            color: customization.isDarkMode ? 'white' : 'inherit',
          }}
        />
      </IconButton>
    </Tooltip>
  );
};

InfoTooltip.propTypes = {
  title: PropTypes.node.isRequired,
};

export default InfoTooltip;

