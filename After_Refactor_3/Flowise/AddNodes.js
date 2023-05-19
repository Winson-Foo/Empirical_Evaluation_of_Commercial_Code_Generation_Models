// AddNodes.js

import { useState, useRef, useEffect } from 'react';
import { useSelector } from 'react-redux';
import PropTypes from 'prop-types';

import PerfectScrollbar from 'react-perfect-scrollbar';

import MainCard from 'ui-component/cards/MainCard';
import Transitions from 'ui-component/extended/Transitions';
import { StyledFab } from 'ui-component/button/StyledFab';
import AccordionList from 'ui-component/AccordionList';
import SearchInput from 'ui-component/SearchInput';

import { IconPlus, IconSearch, IconMinus } from '@tabler/icons';
import { baseURL } from 'store/constant';

const AddNodes = ({ nodesData, selectedNode }) => {
  const theme = useTheme();
  const customization = useSelector((state) => state.customization);

  const [searchValue, setSearchValue] = useState('');
  const [nodesByCategory, setNodesByCategory] = useState({});
  const [isAddNodeOpen, setIsAddNodeOpen] = useState(false);

  const anchorRef = useRef(null);
  const prevIsAddNodeOpen = useRef(isAddNodeOpen);
  const perfectScrollbarRef = useRef();

  const scrollListToTop = () => {
    const listWrapper = perfectScrollbarRef.current;
    if (listWrapper) {
      listWrapper.scrollTop = 0;
    }
  };

  const filterNodesByName = (searchStr) => {
    setSearchValue(searchStr);
    setTimeout(() => {
      if (searchStr) {
        const filteredData = nodesData.filter((node) => {
          return node.name.toLowerCase().includes(searchStr.toLowerCase());
        });
        groupNodesByCategory(filteredData, true);
        scrollListToTop();
      } else if (searchStr === '') {
        groupNodesByCategory(nodesData);
        scrollListToTop();
      }
    }, 500);
  };

  const groupNodesByCategory = (nodes, isFiltered = false) => {
    const nodesByCat = {};
    const isExpandedByCat = {};
    nodes.forEach((node) => {
      if (!nodesByCat[node.category]) {
        nodesByCat[node.category] = [];
        isExpandedByCat[node.category] = isFiltered ? true : false;
      }
      nodesByCat[node.category].push(node);
    });
    setNodesByCategory(nodesByCat);
    setIsCategoryExpanded(isExpandedByCat);
  };

  const handleAccordionChange = (category) => {
    return (event, isExpanded) => {
      const isCategoryExpanded = { ...isExpandedByCat };
      isCategoryExpanded[category] = isExpanded;
      setIsCategoryExpanded(isCategoryExpanded);
    };
  };

  const handleCloseAddNode = (event) => {
    if (anchorRef.current && anchorRef.current.contains(event.target)) {
      return;
    }
    setIsAddNodeOpen(false);
  };

  const handleToggleAddNode = () => {
    setIsAddNodeOpen((prevIsAddNodeOpen) => !prevIsAddNodeOpen);
  };

  const handleNodeDragStart = (event, node) => {
    event.dataTransfer.setData('application/reactflow', JSON.stringify(node));
    event.dataTransfer.effectAllowed = 'move';
  };

  useEffect(() => {
    if (prevIsAddNodeOpen.current === true && isAddNodeOpen === false) {
      anchorRef.current.focus();
    }

    prevIsAddNodeOpen.current = isAddNodeOpen;
  }, [isAddNodeOpen]);

  useEffect(() => {
    if (selectedNode) {
      setIsAddNodeOpen(false);
    }
  }, [selectedNode]);

  useEffect(() => {
    if (nodesData) {
      groupNodesByCategory(nodesData);
    }
  }, [nodesData]);

  return (
    <>
      <StyledFab
        sx={{ left: 20, top: 20 }}
        ref={anchorRef}
        size='small'
        color='primary'
        aria-label='add'
        title='Add Node'
        onClick={handleToggleAddNode}
      >
        {isAddNodeOpen ? <IconMinus /> : <IconPlus />}
      </StyledFab>
      <Popper
        placement='bottom-end'
        open={isAddNodeOpen}
        anchorEl={anchorRef.current}
        role={undefined}
        transition
        disablePortal
        popperOptions={{
          modifiers: [
            {
              name: 'offset',
              options: {
                offset: [-40, 14],
              },
            },
          ],
        }}
        sx={{ zIndex: 1000 }}
      >
        {({ TransitionProps }) => (
          <Transitions in={isAddNodeOpen} {...TransitionProps}>
            <MainCard
              border={false}
              elevation={16}
              content={false}
              boxShadow
              shadow={theme.shadows[16]}
            >
              <Box sx={{ p: 2 }}>
                <Stack>
                  <Typography variant='h4'>Add Nodes</Typography>
                </Stack>
                <SearchInput
                  value={searchValue}
                  onChange={(e) => filterNodesByName(e.target.value)}
                />
                <Divider />
              </Box>
              <PerfectScrollbar
                containerRef={perfectScrollbarRef}
                style={{
                  height: '100%',
                  maxHeight: 'calc(100vh - 320px)',
                  overflowX: 'hidden',
                }}
              >
                <Box sx={{ p: 2 }}>
                  <AccordionList
                    nodesByCategory={nodesByCategory}
                    onNodeDragStart={handleNodeDragStart}
                    isExpandedByCat={isExpandedByCat}
                    onAccordionChange={handleAccordionChange}
                  />
                </Box>
              </PerfectScrollbar>
             </MainCard>
            </ClickAwayListener>
          </Popper>
        )}
      </Box>
    </>
  );
};

AddNodes.propTypes = {
  nodesData: PropTypes.array.isRequired,
  selectedNode: PropTypes.object,
};

export default AddNodes;


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

// SearchInput.js

import React from 'react';
import InputAdornment from '@mui/material/InputAdornment';
import IconSearch from '@tabler/icons';

import OutlinedInput from '@mui/material/OutlinedInput';

const SearchInput = ({ value, onChange }) => {
  const theme = useTheme();
  return (
    <OutlinedInput
      sx={{ width: '100%', pr: 1, pl: 2, my: 2 }}
      id='input-search-node'
      value={value}
      onChange={onChange}
      placeholder='Search nodes'
      startAdornment={
        <InputAdornment position='start'>
          <IconSearch
            stroke={1.5}
            size='1rem'
            color={theme.palette.grey[500]}
          />
        </InputAdornment>
      }
      aria-describedby='search-helper-text'
      inputProps={{
        'aria-label': 'weight',
      }}
    />
  );
};

SearchInput.propTypes = {
  value: PropTypes.string.isRequired,
  onChange: PropTypes.func.isRequired,
};

export default SearchInput;

