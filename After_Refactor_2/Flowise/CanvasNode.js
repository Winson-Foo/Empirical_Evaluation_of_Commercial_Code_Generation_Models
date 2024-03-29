// import statements

import PropTypes from 'prop-types'
import { useContext, useState } from 'react'
import { IconButton, Box, Typography, Divider, Button } from '@mui/material'

// custom components
import MainCard from 'ui-component/cards/MainCard'
import NodeInputHandler from './NodeInputHandler'
import NodeOutputHandler from './NodeOutputHandler'
import AdditionalParamsDialog from 'ui-component/dialog/AdditionalParamsDialog'

// styles
import { cardWrapperStyles } from './canvasNodeStyles'

// const
import { baseURL } from 'store/constant'
import { IconTrash, IconCopy } from '@tabler/icons'
import { flowContext } from 'store/context/ReactFlowContext'

const CanvasNode = ({ data }) => {
    const theme = useTheme()
    const { deleteNode, duplicateNode } = useContext(flowContext)

    const [showDialog, setShowDialog] = useState(false)
    const [dialogProps, setDialogProps] = useState({})

    const onAdditionalParamsClicked = () => {
        const dialogProps = {
            data,
            inputParams: data.inputParams.filter((param) => param.additionalParams),
            confirmButtonName: 'Save',
            cancelButtonName: 'Cancel'
        }
        setDialogProps(dialogProps)
        setShowDialog(true)
    }

    return (
        <>
            <MainCard
                content={false}
                sx={{
                    padding: 0,
                    borderColor: data.selected ? theme.palette.primary.main : theme.palette.text.secondary,
                    ...cardWrapperStyles(theme.palette.card.main, theme.darkTextPrimary, theme.palette.primary),
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
                        <div style={{ flexGrow: 1 }}></div>
                        <IconButton
                            title='Duplicate'
                            onClick={() => {
                                duplicateNode(data.id)
                            }}
                            sx={{ height: 35, width: 35, '&:hover': { color: theme?.palette.primary.main } }}
                            color={theme?.customization?.isDarkMode ? theme.colors?.paper : 'inherit'}
                        >
                            <IconCopy />
                        </IconButton>
                        <IconButton
                            title='Delete'
                            onClick={() => {
                                deleteNode(data.id)
                            }}
                            sx={{ height: 35, width: 35, mr: 1, '&:hover': { color: 'red' } }}
                            color={theme?.customization?.isDarkMode ? theme.colors?.paper : 'inherit'}
                        >
                            <IconTrash />
                        </IconButton>
                    </div>
                    {(data.inputAnchors.length > 0 || data.inputParams.length > 0) && (
                        <>
                            <Divider />
                            <Box sx={{ background: theme.palette.asyncSelect.main, p: 1 }}>
                                <Typography fontWeight={500} textAlign='center' variant='body1'>
                                    Inputs
                                </Typography>
                            </Box>
                            <Divider />
                        </>
                    )}
                    {data.inputAnchors.map((inputAnchor, index) => (
                        <NodeInputHandler key={index} inputAnchor={inputAnchor} data={data} />
                    ))}
                    {data.inputParams.map((inputParam, index) => (
                        <NodeInputHandler key={index} inputParam={inputParam} data={data} />
                    ))}
                    {data.inputParams.find((param) => param.additionalParams) && (
                        <div style={{ textAlign: 'center' }}>
                            <Button sx={{ borderRadius: 25, width: '90%', mb: 2 }} variant='outlined' onClick={onAdditionalParamsClicked}>
                                Additional Parameters
                            </Button>
                        </div>
                    )}
                    <Divider />
                    <Box sx={{ background: theme.palette.asyncSelect.main, p: 1 }}>
                        <Typography fontWeight={500} textAlign='center' variant='body1'>
                            Output
                        </Typography>
                    </Box>
                    <Divider />

                    {data.outputAnchors.map((outputAnchor, index) => (
                        <NodeOutputHandler key={index} outputAnchor={outputAnchor} data={data} />
                    ))}
                </Box>
            </MainCard>
            <AdditionalParamsDialog
                show={showDialog}
                dialogProps={dialogProps}
                onCancel={() => setShowDialog(false)}
            ></AdditionalParamsDialog>
        </>
    )
}

CanvasNode.propTypes = {
    data: PropTypes.object
}

export default CanvasNode

// canvasNodeStyles.js
export const cardWrapperStyles = (background, color, primary) => ({
    background,
    color,
    border: `solid 1px ${primary[200]} 75`,
    width: '300px',
    height: 'auto',
    padding: '10px',
    boxShadow: '0 2px 14px 0 rgb(32 40 45 / 8%)',
    '&:hover': {
        borderColor: primary.main,
    }
})

