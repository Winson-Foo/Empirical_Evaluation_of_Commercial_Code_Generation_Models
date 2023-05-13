const getHandle = (id, position, isValidConnection) => (
    <CustomWidthTooltip placement='left' title={inputAnchor.type}>
        <Handle
            type='target'
            position={Position.Left}
            key={id}
            id={id}
            isValidConnection={isValidConnection}
            style={{
                height: 10,
                width: 10,
                backgroundColor: selected ? theme.palette.primary.main : theme.palette.text.secondary,
                top: position,
            }}
        />
    </CustomWidthTooltip>
)

const getLabel = (label, optional) => (
    <Typography>
        {label}
        {!optional && <span style={{ color: 'red' }}>&nbsp;*</span>}
    </Typography>
)

const NodeInputHandler = ({ inputAnchor, inputParam, data, disabled = false, isAdditionalParams = false }) => {
    const { reactFlowInstance } = useContext(flowContext)
    const customization = useSelector((state) => state.customization)

    const theme = useTheme()

    const ref = useRef(null)
    const updateNodeInternals = useUpdateNodeInternals()

    const [position, setPosition] = useState(0)
    const [showExpandDialog, setShowExpandDialog] = useState(false)
    const [expandDialogProps, setExpandDialogProps] = useState({})

    const { id, selected, inputs } = data

    useEffect(() => {
        if (ref.current) {
            setPosition(ref.current.offsetTop + ref.current.clientHeight / 2)
            updateNodeInternals(id)
        }
    }, [id, ref, updateNodeInternals])

    useEffect(() => {
        updateNodeInternals(id)
    }, [id, position, updateNodeInternals])

    const handleExpandDialogClicked = () => {
        const dialogProp = {
            value: inputs[inputParam.name] ?? inputParam.default ?? '',
            inputParam,
            disabled,
            confirmButtonName: 'Save',
            cancelButtonName: 'Cancel',
        }

        if (!disabled) {
            const nodes = reactFlowInstance.getNodes()
            const edges = reactFlowInstance.getEdges()
            const nodesForVariable = inputParam.acceptVariable ? getAvailableNodesForVariable(nodes, edges, id, inputParam.id) : []
            dialogProp.availableNodesForVariable = nodesForVariable
        }

        setExpandDialogProps(dialogProp)
        setShowExpandDialog(true)
    }

    const handleExpandDialogSave = (newValue, inputParamName) => {
        setShowExpandDialog(false)
        inputs[inputParamName] = newValue
    }

    return (
        <div ref={ref}>
            {inputAnchor && (
                <>
                    {getHandle(inputAnchor.id, position, (connection) => isValidConnection(connection, reactFlowInstance))}
                    <Box sx={{ p: 2 }}>{getLabel(inputAnchor.label, inputAnchor.optional)}</Box>
                </>
            )}

            {((inputParam && !inputParam.additionalParams) || isAdditionalParams) && (
                <>
                    {inputParam.acceptVariable && getHandle(inputParam.id, position, (connection) => isValidConnection(connection, reactFlowInstance))}
                    <Box sx={{ p: 2 }}>
                        <div style={{ display: 'flex', flexDirection: 'row' }}>
                            {getLabel(inputParam.label, inputParam.optional)}
                            <div style={{ flexGrow: 1 }}></div>
                            {inputParam.type === 'string' && inputParam.rows && (
                                <IconButton
                                    size='small'
                                    sx={{ height: 25, width: 25 }}
                                    title='Expand'
                                    color='primary'
                                    onClick={handleExpandDialogClicked}
                                >
                                    <IconArrowsMaximize />
                                </IconButton>
                            )}
                        </div>
                        {inputParam.type === 'file' && (
                            <File
                                disabled={disabled}
                                fileType={inputParam.fileType || '*'}
                                onChange={(newValue) => (inputs[inputParam.name] = newValue)}
                                value={inputs[inputParam.name] ?? inputParam.default ?? 'Choose a file to upload'}
                            />
                        )}
                        {inputParam.type === 'boolean' && (
                            <SwitchInput
                                disabled={disabled}
                                onChange={(newValue) => (inputs[inputParam.name] = newValue)}
                                value={inputs[inputParam.name] ?? inputParam.default ?? false}
                            />
                        )}
                        {(inputParam.type === 'string' || inputParam.type === 'password' || inputParam.type === 'number') && (
                            <Input
                                disabled={disabled}
                                inputParam={inputParam}
                                onChange={(newValue) => (inputs[inputParam.name] = newValue)}
                                value={inputs[inputParam.name] ?? inputParam.default ?? ''}
                                showDialog={showExpandDialog}
                                dialogProps={expandDialogProps}
                                onDialogCancel={() => setShowExpandDialog(false)}
                                onDialogConfirm={(newValue, inputParamName) => handleExpandDialogSave(newValue, inputParamName)}
                            />
                        )}
                        {inputParam.type === 'json' && (
                            <JsonEditorInput
                                disabled={disabled}
                                onChange={(newValue) => (inputs[inputParam.name] = newValue)}
                                value={inputs[inputParam.name] ?? inputParam.default ?? ''}
                                isDarkMode={customization.isDarkMode}
                            />
                        )}
                        {inputParam.type === 'options' && (
                            <Dropdown
                                disabled={disabled}
                                name={inputParam.name}
                                options={inputParam.options}
                                onSelect={(newValue) => (inputs[inputParam.name] = newValue)}
                                value={inputs[inputParam.name] ?? inputParam.default ?? 'chose an option'}
                            />
                        )}
                    </Box>
                </>
            )}
        </div>
    )
}

NodeInputHandler.propTypes = {
    inputAnchor: PropTypes.object,
    inputParam: PropTypes.object,
    data: PropTypes.object,
    disabled: PropTypes.bool,
    isAdditionalParams: PropTypes.bool,
}

const NodeInputHandle = ({ id, position, isValidConnection, selected }) => {
    const { reactFlowInstance } = useContext(flowContext)
    const theme = useTheme()

    return (
        <CustomWidthTooltip placement='left' title={inputAnchor.type}>
            <Handle
                type='target'
                position={Position.Left}
                key={id}
                id={id}
                isValidConnection={isValidConnection}
                style={{
                    height: 10,
                    width: 10,
                    backgroundColor: selected ? theme.palette.primary.main : theme.palette.text.secondary,
                    top: position,
                }}
            />
        </CustomWidthTooltip>
    )
}

const RequiredLabel = ({ label }) => (
    <Typography>
        {label}
        <span style={{ color: 'red' }}>&nbsp;*</span>
    </Typography>
)

const OptionalLabel = ({ label }) => (
    <Typography>
        {label}
        &nbsp;(Optional)
    </Typography>
)

const NodeInputLabel = ({ label, optional }) => {
    const LabelComponent = optional ? OptionalLabel : RequiredLabel
    return <LabelComponent label={label} />
}

const NodeInputHandler = ({ inputAnchor, inputParam, data, disabled = false, isAdditionalParams = false }) => {
    const customization = useSelector((state) => state.customization)
    const { reactFlowInstance } = useContext(flowContext)

    const ref = useRef(null)
    const updateNodeInternals = useUpdateNodeInternals()

    const [position, setPosition] = useState(0)
    const [showExpandDialog, setShowExpandDialog] = useState(false)
    const [expandDialogProps, setExpandDialogProps] = useState({})

    const { id, inputs } = data

    useEffect(() => {
        if (ref.current) {
            setPosition(ref.current.offsetTop + ref.current.clientHeight / 2)
            updateNodeInternals(id)
        }
    }, [id, ref, updateNodeInternals])

    useEffect(() => {
        updateNodeInternals(id)
    }, [id, position, updateNodeInternals])

    const handleExpandDialogClicked = () => {
        const dialogProp = {
            value: inputs[inputParam.name] ?? inputParam.default ?? '',
            inputParam,
            disabled,
            confirmButtonName: 'Save',
            cancelButtonName: 'Cancel',
        }

        if (!disabled) {
            const nodes = reactFlowInstance.getNodes()
            const edges = reactFlowInstance.getEdges()
            const nodesForVariable = inputParam.acceptVariable ? getAvailableNodesForVariable(nodes, edges, id, inputParam.id) : []
            dialogProp.availableNodesForVariable = nodesForVariable
        }

        setExpandDialogProps(dialogProp)
        setShowExpandDialog(true)
    }

    const handleExpandDialogSave = (newValue, inputParamName) => {
        setShowExpandDialog(false)
        inputs[inputParamName] = newValue
    }

    const inputLabel = <NodeInputLabel label={inputParam.label} optional={inputParam.optional} />

    return (
        <div ref={ref}>
            {inputAnchor && (
                <>
                    <NodeInputHandle
                        id={inputAnchor.id}
                        position={position}
                        selected={data.selected}
                        isValidConnection={(connection) => isValidConnection(connection, reactFlowInstance)}
                    />
                    <Box sx={{ p: 2 }}>{inputLabel}</Box>
                </>
            )}

            {((inputParam && !inputParam.additionalParams) || isAdditionalParams) && (
                <>
                    {inputParam.acceptVariable && (
                        <NodeInputHandle
                            id={inputParam.id}
                            position={position}
                            selected={data.selected}
                            isValidConnection={(connection) => isValidConnection(connection, reactFlowInstance)}
                        />
                    )}

                    <Box sx={{ p: 2 }}>
                        <div style={{ display: 'flex', flexDirection: 'row' }}>
                            {inputLabel}
                            <div style={{ flexGrow: 1 }}></div>
                            {inputParam.type === 'string' && inputParam.rows && (
                                <IconButton
                                    size='small'
                                    sx={{ height: 25, width: 25 }}
                                    title='Expand'
                                    color='primary'
                                    onClick={handleExpandDialogClicked}
                                >
                                    <IconArrowsMaximize />
                                </IconButton>
                            )}
                        </div>
                        {inputParam.type === 'file' && (
                            <File
                                disabled={disabled}
                                fileType={inputParam.fileType || '*'}
                                onChange={(newValue) => (inputs[inputParam.name] = newValue)}
                                value={inputs[inputParam.name] ?? inputParam.default ?? 'Choose a file to upload'}
                            />
                        )}
                        {inputParam.type === 'boolean' && (
                            <SwitchInput
                                disabled={disabled}
                                onChange={(newValue) => (inputs[inputParam.name] = newValue)}
                                value={inputs[inputParam.name] ?? inputParam.default ?? false}
                            />
                        )}
                        {(inputParam.type === 'string' || inputParam.type === 'password' || inputParam.type === 'number') && (
                            <Input
                                disabled={disabled}
                                inputParam={inputParam}
                                onChange={(newValue) => (inputs[inputParam.name] = newValue)}
                                value={inputs[inputParam.name] ?? input

