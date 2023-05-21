// AccordionList.js

import React from 'react';
import PropTypes from 'prop-types';

import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import ListItemButton from '@mui/material/ListItemButton';
import ListItem from '@mui/material/ListItem';
import ListItemAvatar from '@mui/material/ListItemAvatar';
import ListItemText from '@mui/material/ListItemText';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';

import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

const AccordionList = ({
  nodesByCategory,
  onNodeDragStart,
  isExpandedByCat,
  onAccordionChange,
}) => {
  return (
    <List
      sx={{
        width: '100%',
        maxWidth: 370,
        py: 0,
        borderRadius: '10px',
        [theme.breakpoints.down('md')]: {
          maxWidth: 370,
        },
        '& .MuiListItemSecondaryAction-root': {
          top: 22,
        },
        '& .MuiDivider-root': {
          my: 0,
        },
        '& .list-container': {
          pl: 7,
        },
      }}
    >
      {Object.keys(nodesByCategory)
        .sort()
        .map((category) => (
          <Accordion
            expanded={isExpandedByCat[category] || false}
            onChange={onAccordionChange(category)}
            key={category}
          >
            <AccordionSummary
              expandIcon={<ExpandMoreIcon />}
              aria-controls={`nodes-accordion-${category}`}
              id={`nodes-accordion-header-${category}`}
            >
              <Typography variant='h5'>{category}</Typography>
            </AccordionSummary>
            <AccordionDetails>
              {nodesByCategory[category].map((node) => (
                <div
                  key={node.name}
                  onDragStart={(event) => onNodeDragStart(event, node)}
                  draggable
                >
                  <ListItemButton
                    sx={{
                      p: 0,
                      borderRadius: `${customization.borderRadius}px`,
                      cursor: 'move',
                    }}
                  >
                    <ListItem alignItems='center'>
                      <ListItemAvatar>
                        <div
                          style={{
                            width: 50,
                            height: 50,
                            borderRadius: '50%',
                            backgroundColor: 'white',
                          }}
                        >
                          <img
                            style={{
                              width: '100%',
                              height: '100%',
                              padding: 10,
                              objectFit: 'contain',
                            }}
                            alt={node.name}
                            src={`${baseURL}/api/v1/node-icon/${node.name}`}
                          />
                        </div>
                      </ListItemAvatar>
                      <ListItemText
                        sx={{ ml: 1 }}
                        primary={node.label}
                        secondary={node.description}
                      />
                    </ListItem>
                  </ListItemButton>
                  <Divider />
                </div>
              ))}
            </AccordionDetails>
          </Accordion>
        ))}
    </List>
  );
};

AccordionList.propTypes = {
  nodesByCategory: PropTypes.object.isRequired,
  onNodeDragStart: PropTypes.func,
  isExpandedByCat: PropTypes.object.isRequired,
  onAccordionChange: PropTypes.func.isRequired,
};

export default AccordionList;