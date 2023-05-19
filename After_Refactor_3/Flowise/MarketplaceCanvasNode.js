// styles.js
import { styled } from '@mui/material/styles'
import MainCard from 'ui-component/cards/MainCard'

export const CardWrapper = styled(MainCard)(({ theme }) => ({
    background: theme.palette.card.main,
    color: theme.darkTextPrimary,
    border: 'solid 1px',
    borderColor: theme.palette.primary[200] + 75,
    width: '300px',
    height: 'auto',
    padding: '10px',
    boxShadow: '0 2px 14px 0 rgb(32 40 45 / 8%)',
    '&:hover': {
        borderColor: theme.palette.primary.main
    }
}))

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

// AdditionalParamsDialog.js
import PropTypes from 'prop-types'
import { useState } from 'react'
import { Box, Typography, Button, Dialog } from '@mui/material'

const AdditionalParamsDialog = ({ show, dialogProps, onCancel }) => {
    const [isOpen, setIsOpen] = useState(show)

    return (
        <Dialog open={isOpen}>
            <Box sx={{ p: 3 }}>
                <Typography variant='h6' sx={{ mb: 3 }}>
                    Additional Parameters
                </Typography>
                {/* render additional params */}
                <Button variant='outlined' onClick={onCancel}>
                    Close
                </Button>
            </Box>
        </Dialog>
    )
}

AdditionalParamsDialog.propTypes = {
    show: PropTypes.bool,
    dialogProps: PropTypes.object,
    onCancel: PropTypes.func
}

export default AdditionalParamsDialog

// MarketplaceCanvasNode.js
import PropTypes from 'prop-types'
import { useState } from 'react'
import { CardWrapper } from './styles'
import { Box, Typography, Divider, Button } from '@mui/material'
import NodeInputs from './NodeInputs'
import NodeOutputs from './NodeOutputs'
import AdditionalParamsDialog from 'ui-component/dialog/AdditionalParamsDialog'
import { baseURL } from 'store/constant'

const MarketplaceCanvasNode = ({ data }) => {
    const theme = useTheme()

    const [showDialog, setShowDialog] = useState(false)
    const [dialogProps, setDialogProps] = useState({})

    const onDialogClicked = () => {
        const dialogProps = {
            data,
            inputParams: data.inputParams.filter((param) => param.additionalParams),
            disabled: true,
            confirmButtonName: 'Save',
            cancelButtonName: 'Cancel'
        }
        setDialogProps(dialogProps)
        setShowDialog(true)
    }

    return (
        <>
            <CardWrapper
                content={false}
                sx={{
                    padding: 0,
                    borderColor: data.selected ? theme.palette.primary.main : theme.palette.text.secondary
                }}
                border={false}
            >
                <Box>
                    <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
                        <Box style={{ width: 50, marginRight: 10, padding: 5 }}>
                            <div
                                style={{
                                    ...theme.typography.commonAvatar,
                                    ...theme.typography.largeAvatar,
                                    borderRadius: '50%',
                                    backgroundColor: 'white',
                                    cursor: 'grab'
                                }}
                            >
                                <img
                                    style={{ width: '100%', height: '100%', padding: 5, objectFit: 'contain' }}
                                    src={`${baseURL}/api/v1/node-icon/${data.name}`}
                                    alt='Notification'
                                />
                            </div>
                        </Box>
                        <Box>
                            <Typography
                                sx={{
                                    fontSize: '1rem',
                                    fontWeight: 500
                                }}
                            >
                                {data.label}
                            </Typography>
                        </Box>
                    </div>
                    <NodeInputs data={data} />
                    <NodeOutputs data={data} />
                </Box>
            </CardWrapper>
            <AdditionalParamsDialog
                show={showDialog}
                dialogProps={dialogProps}
                onCancel={() => setShowDialog(false)}
            ></AdditionalParamsDialog>
        </>
    )
}

MarketplaceCanvasNode.propTypes = {
    data: PropTypes.object
}

export default MarketplaceCanvasNode

