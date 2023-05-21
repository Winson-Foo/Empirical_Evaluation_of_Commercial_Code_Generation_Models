// NodeInputs.js
import PropTypes from 'prop-types'
import NodeInputHandler from 'views/canvas/NodeInputHandler'

const NodeInputs = ({ data }) => {
    const inputParams = data.inputParams.filter((param) => param.additionalParams)

    return (
        <>
            {data.inputAnchors.length > 0 || inputParams.length > 0 && (
                <>
                    <Divider />
                    <Box sx={{ background: theme.palette.asyncSelect.main, p: 1 }}>
                        <Typography
                            sx={{
                                fontWeight: 500,
                                textAlign: 'center'
                            }}
                        >
                            Inputs
                        </Typography>
                    </Box>
                    <Divider />
                </>
            )}
            {data.inputAnchors.map((inputAnchor, index) => (
                <NodeInputHandler disabled={true} key={index} inputAnchor={inputAnchor} data={data} />
            ))}
            {inputParams.map((inputParam, index) => (
                <NodeInputHandler disabled={true} key={index} inputParam={inputParam} data={data} />
            ))}
            {inputParams.length > 0 && (
                <div style={{ textAlign: 'center' }}>
                    <Button sx={{ borderRadius: 25, width: '90%', mb: 2 }} variant='outlined' onClick={onDialogClicked}>
                        Additional Parameters
                    </Button>
                </div>
            )}
        </>
    )
}

NodeInputs.propTypes = {
    data: PropTypes.object
}

export default NodeInputs