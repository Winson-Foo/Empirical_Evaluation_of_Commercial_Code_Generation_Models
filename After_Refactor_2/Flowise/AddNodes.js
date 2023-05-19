import React, { useState, useRef, useEffect } from 'react';
import PropTypes from 'prop-types';
import { useSelector } from 'react-redux';

import AddNodeButton from './AddNodeButton';
import NodeList from './NodeList';

import { getNodesByCategory } from '../../../utils/nodes';

import { StyledFab } from 'ui-component/button/StyledFab';
import { IconPlus, IconSearch, IconMinus } from '@tabler/icons';

import { useTheme } from '@mui/material/styles';

import {
  Box,
  ClickAwayListener,
  Divider,
  InputAdornment,
  OutlinedInput,
  Popper,
  Stack,
  Typography,
} from '@mui/material';

import Transitions from 'ui-component/extended/Transitions';
import MainCard from 'ui-component/cards/MainCard';

const NodeSelector = ({ onSelectNode, selectedNodeId }) => {
  const theme = useTheme();
  const customization = useSelector((state) => state.customization);
  const [searchValue, setSearchValue] = useState('');
  const [open, setOpen] = useState(false);
  const anchorRef = useRef(null);
  const prevOpen = useRef(open);
  const [categories, setCategories] = useState([]);
  const [nodesByCategory, setNodesByCategory] = useState({});
  const ps = useRef();

  const handleToggle = () => {
    setOpen((prevOpen) => !prevOpen);
  };

  const handleClose = (event) => {
    if (anchorRef.current && anchorRef.current.contains(event.target)) {
      return;
    }
    setOpen(false);
  };

  const filterSearch = (value) => {
    setSearchValue(value);
  };

  const handleCategorySelected = (categoryId) => {
    const selectedNode = nodesByCategory[categoryId].find(
      (node) => node.id === selectedNodeId
    );
    if (selectedNode) {
      onSelectNode(selectedNode);
    } else {
      onSelectNode(nodesByCategory[categoryId][0]);
    }
  };

  useEffect(() => {
    const nodesByCategory = getNodesByCategory();
    setNodesByCategory(nodesByCategory);
    setCategories(Object.keys(nodesByCategory));
    if (selectedNodeId) {
      const category = Object.keys(nodesByCategory).find((categoryId) =>
        nodesByCategory[categoryId].some((node) => node.id === selectedNodeId)
      );
      if (category) handleCategorySelected(category);
    }
  }, [selectedNodeId]);

  useEffect(() => {
    if (prevOpen.current === true && open === false) {
      anchorRef.current.focus();
    }
    prevOpen.current = open;
  }, [open]);

  return (
    <>
      <AddNodeButton handleClick={handleToggle} />
      <Popper
        placement='bottom-end'
        open={open}
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
          <Transitions in={open} {...TransitionProps}>
            <MainCard
              border={false}
              elevation={16}
              content={false}
              boxShadow
              shadow={theme.shadows[16]}
            >
              <ClickAwayListener onClickAway={handleClose}>
                <Box sx={{ p: 2 }}>
                  <Stack>
                    <Typography variant='h4'>Add Nodes</Typography>
                  </Stack>
                  <OutlinedInput
                    sx={{ width: '100%', pr: 1, pl: 2, my: 2 }}
                    id='input-search-node'
                    value={searchValue}
                    onChange={(e) => filterSearch(e.target.value)}
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
                  <Divider />
                </Box>
                <Box
                  sx={{
                    maxHeight: 'calc(100vh - 320px)',
                    overflowY: 'scroll',
                  }}
                  ref={ps}
                >
                  <NodeList
                    categories={categories}
                    nodesByCategory={nodesByCategory}
                    handleCategorySelected={handleCategorySelected}
                    selectedNodeId={selectedNodeId}
                    searchValue={searchValue}
                    onSelectNode={onSelectNode}
                  />
                </Box>
              </ClickAwayListener>
            </MainCard>
          </Transitions>
        )}
      </Popper>
    </>
  );
};

NodeSelector.propTypes = {
  onSelectNode: PropTypes.func.isRequired,
  selectedNodeId: PropTypes.string,
};

export default NodeSelector;