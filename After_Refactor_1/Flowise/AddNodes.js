// NODES LIST COMPONENT
const AccordionSection = ({ category, nodes, expandedCategories, handleAccordionChange, customization }) => (
  <Accordion
    expanded={expandedCategories[category] || false}
    onChange={handleAccordionChange(category)}
    key={category}
  >
    <AccordionSummary
      expandIcon={<ExpandMoreIcon />}
      aria-controls={`nodes-accordian-${category}`}
      id={`nodes-accordian-header-${category}`}
    >
      <Typography variant='h5'>{category}</Typography>
    </AccordionSummary>
    <AccordionDetails>
      {nodes.map((node, index) => (
        <NodeListItem
          key={node.name}
          node={node}
          customization={customization}
          onDragStart={(event) => onDragStart(event, node)}
          isLast={index === nodes.length - 1}
        />
      ))}
    </AccordionDetails>
  </Accordion>
);

const NodeListItem = ({ node, customization, onDragStart, isLast }) => (
  <div onDragStart={onDragStart} draggable>
    <ListItemButton sx={{ p: 0, borderRadius: `${customization.borderRadius}px`, cursor: 'move' }}>
      <ListItem alignItems='center'>
        <ListItemAvatar>
          <div style={{ width: 50, height: 50, borderRadius: '50%', backgroundColor: 'white' }}>
            <img
              style={{ width: '100%', height: '100%', padding: 10, objectFit: 'contain' }}
              alt={node.name}
              src={`${baseURL}/api/v1/node-icon/${node.name}`}
            />
          </div>
        </ListItemAvatar>
        <ListItemText sx={{ ml: 1 }} primary={node.label} secondary={node.description} />
      </ListItem>
    </ListItemButton>
    {isLast ? null : <Divider />}
  </div>
);

const SearchBox = ({ searchValue, onChangeSearchValue }) => (
  <OutlinedInput
    sx={{ width: '100%', pr: 1, pl: 2, my: 2 }}
    id='input-search-node'
    value={searchValue}
    onChange={(e) => onChangeSearchValue(e.target.value)}
    placeholder='Search nodes'
    startAdornment={
      <InputAdornment position='start'>
        <IconSearch stroke={1.5} size='1rem' color={theme.palette.grey[500]} />
      </InputAdornment>
    }
    aria-describedby='search-helper-text'
    inputProps={{ 'aria-label': 'weight' }}
  />
);

const NodesList = ({ nodesData }) => {
  const [searchValue, setSearchValue] = useState('');
  const [nodesByCategory, setNodesByCategory] = useState({});
  const [searchedNodes, setSearchedNodes] = useState([]);
  const [open, setOpen] = useState(false);
  const [expandedCategories, setExpandedCategories] = useState({});
  const anchorRef = useRef(null);
  const prevOpen = useRef(open);
  const ps = useRef();

  const theme = useTheme();
  const customization = useSelector((state) => state.customization);

  const scrollTop = () => {
    const curr = ps.current;
    if (curr) {
      curr.scrollTop = 0;
    }
  };

  const filterNodes = () => {
    const returnData = nodesData.filter((nd) =>
      nd.name.toLowerCase().includes(searchValue.toLowerCase())
    );
    setSearchedNodes(returnData);
    groupNodesByCategory(returnData, true);
    scrollTop();
  };

  const onChangeSearchValue = (value) => {
    setSearchValue(value);
    setTimeout(() => {
      if (value) {
        filterNodes();
      } else if (value === '') {
        groupNodesByCategory(nodesData);
        setSearchedNodes([]);
        scrollTop();
      }
    }, 500);
  };

  const groupNodesByCategory = (nodes, isFilter = false) => {
    const newNodeDict = {};
    const newCategoryDict = {};
    nodes.forEach((node) => {
      if (!newNodeDict[node.category]) {
        newNodeDict[node.category] = [node];
        newCategoryDict[node.category] = isFilter ? true : false;
      } else {
        newNodeDict[node.category].push(node);
      }
    });
    setNodesByCategory(newNodeDict);
    setExpandedCategories(newCategoryDict);
  };

  const handleAccordionChange = (category) => (e, isExpanded) => {
    const newExpandedCategories = { ...expandedCategories };
    newExpandedCategories[category] = isExpanded;
    setExpandedCategories(newExpandedCategories);
  };

  const handleClose = (event) => {
    if (anchorRef.current && anchorRef.current.contains(event.target)) {
      return;
    }

    setOpen(false);
  };

  const handleToggle = () => {
    setOpen((prevOpen) => !prevOpen);
  };

  const onDragStart = (event, node) => {
    event.dataTransfer.setData('application/reactflow', JSON.stringify(node));
    event.dataTransfer.effectAllowed = 'move';
  };

  useEffect(() => {
    if (prevOpen.current === true && open === false) {
      anchorRef.current.focus();
    }

    prevOpen.current = open;
  }, [open]);

  useEffect(() => {
    groupNodesByCategory(nodesData);
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
        onClick={handleToggle}
      >
        {open ? <IconMinus /> : <IconPlus />}
      </StyledFab>
      <Popper
        placement='bottom-end'
        open={open}
        anchorEl={anchorRef.current}
        role={undefined}
        transition
        disablePortal
        popperOptions={{
          modifiers: [
            { name: 'offset', options: { offset: [-40, 14] } },
          ],
        }}
        sx={{ zIndex: 1000 }}
      >
        {({ TransitionProps }) => (
          <Transitions in={open} {...TransitionProps}>
            <Paper>
              <ClickAwayListener onClickAway={handleClose}>
                <MainCard border={false} elevation={16} content={false} boxShadow shadow={theme.shadows[16]}>
                  <Box sx={{ p: 2 }}>
                    <Stack>
                      <Typography variant='h4'>Add Nodes</Typography>
                    </Stack>
                    <SearchBox
                      searchValue={searchValue}
                      onChangeSearchValue={onChangeSearchValue}
                    />
                    <Divider />
                  </Box>
                  <PerfectScrollbar
                    containerRef={(el) => { ps.current = el }}
                    style={{ height: '100%', maxHeight: 'calc(100vh - 320px)', overflowX: 'hidden' }}
                  >
                    <Box sx={{ p: 2 }}>
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
                        {searchValue
                          ? searchedNodes.map((node) => (
                            <NodeListItem
                              key={node.name}
                              node={node}
                              customization={customization}
                              onDragStart={(event) => onDragStart(event, node)}
                              isLast={searchedNodes.indexOf(node) === searchedNodes.length - 1}
                            />
                          ))
                          : Object.entries(nodesByCategory)
                            .sort()
                            .map(([category, nodes]) => (
                              <AccordionSection
                                key={category}
                                category={category}
                                nodes={nodes}
                                expandedCategories={expandedCategories}
                                handleAccordionChange={handleAccordionChange}
                                customization={customization}
                              />
                            ))}
                      </List>
                    </Box>
                  </PerfectScrollbar>
                </MainCard>
              </ClickAwayListener>
            </Paper>
          </Transitions>
        )}
      </Popper>
    </>
  );
};

NodesList.propTypes = {
  nodesData: PropTypes.array,
};

export default NodesList;

