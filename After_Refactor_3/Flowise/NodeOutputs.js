// NodeOutputs.js
import NodeOutputHandler from 'views/canvas/NodeOutputHandler'

const NodeOutputs = ({ data }) => {
    return (
        <>
            <Divider />
            <Box sx={{ background: theme.palette.asyncSelect.main, p: 1 }}>
                <Typography
                    sx={{
                        fontWeight: 500,
                        textAlign: 'center'
                    }}
                >
                    Output
                </Typography>
            </Box>
            <Divider />
            {data.outputAnchors.map((outputAnchor, index) => (
                <NodeOutputHandler disabled={true} key={index} outputAnchor={outputAnchor} data={data} />
            ))}
        </>
    )
}

NodeOutputs.propTypes = {
    data: PropTypes.object
}

export default NodeOutputs