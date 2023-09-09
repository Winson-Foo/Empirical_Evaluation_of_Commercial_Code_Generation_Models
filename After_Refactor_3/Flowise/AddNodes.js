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