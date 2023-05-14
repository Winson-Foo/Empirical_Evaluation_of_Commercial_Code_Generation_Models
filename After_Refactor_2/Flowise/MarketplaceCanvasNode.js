// MarketplaceCanvasNode.js

import PropTypes from 'prop-types'
import { useState } from 'react'
import NodeInputHandler from 'views/canvas/NodeInputHandler'
import NodeOutputHandler from 'views/canvas/NodeOutputHandler'
import AdditionalParamsDialog from 'ui-component/dialog/AdditionalParamsDialog'
import { baseAPIURL } from 'store/constant'
import { CardWrapper, NodeContainer, NodeImage, NodeLabel, SectionHeader,
         InputContainer, OutputContainer, AdditionalParamsButton } from './styles'

const MarketplaceCanvasNode = ({ data }) => {
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
            <CardWrapper selected={data.selected}>
                <NodeContainer>
                    <NodeImage src={`${baseAPIURL}/api/v1/node-icon/${data.name}`} alt='Notification' />
                    <NodeLabel>{data.label}</NodeLabel>
                </NodeContainer>
                <SectionHeader>Inputs</SectionHeader>
                <InputContainer>
                    {data.inputAnchors.map((inputAnchor, index) => (
                        <NodeInputHandler disabled={true} key={index} inputAnchor={inputAnchor} data={data} />
                    ))}
                    {data.inputParams.map((inputParam, index) => (
                        <NodeInputHandler disabled={true} key={index} inputParam={inputParam} data={data} />
                    ))}
                    {data.inputParams.find((param) => param.additionalParams) && (
                        <AdditionalParamsButton 
                            width='90%' 
                            mb='2' 
                            variant='outlined' 
                            onClick={onDialogClicked}
                        >
                            Additional Parameters
                        </AdditionalParamsButton>
                    )}
                </InputContainer>
                <SectionHeader>Output</SectionHeader>
                <OutputContainer>
                    {data.outputAnchors.map((outputAnchor, index) => (
                        <NodeOutputHandler disabled={true} key={index} outputAnchor={outputAnchor} data={data} />
                    ))}
                </OutputContainer>
            </CardWrapper>
            <AdditionalParamsDialog
                show={showDialog}
                dialogProps={dialogProps}
                onCancel={() => setShowDialog(false)}
            />
        </>
    )
}

MarketplaceCanvasNode.propTypes = {
    data: PropTypes.object.isRequired
}

export default MarketplaceCanvasNode


// styles.js

import { styled } from '@mui/material/styles'
import { Box, Typography, Button } from '@mui/material'

export const CardWrapper = styled(Box)(({ theme, selected }) => ({
    background: theme.palette.card.main,
    color: theme.darkTextPrimary,
    border: 'solid 1px',
    borderColor: selected ? theme.palette.primary.main : theme.palette.text.secondary,
    width: '300px',
    height: 'auto',
    padding: '10px',
    boxShadow: '0 2px 14px 0 rgb(32 40 45 / 8%)',
    '&:hover': {
        borderColor: theme.palette.primary.main
    }
}))

export const NodeContainer = styled(Box)({
    display: 'flex',
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: '10px'
})

export const NodeImage = styled('img')({
    ...theme.typography.commonAvatar,
    ...theme.typography.largeAvatar,
    borderRadius: '50%',
    backgroundColor: 'white',
    cursor: 'grab',
    width: '50px',
    height: '50px',
    padding: '5px',
    objectFit: 'contain'
})

export const NodeLabel = styled(Typography)({
    fontSize: '1rem',
    fontWeight: 500,
    marginLeft: '10px'
})

export const SectionHeader = styled(Typography)({
    fontWeight: 500,
    textAlign: 'center',
    margin: '10px 0px'
})

export const InputContainer = styled(Box)({
    background: theme.palette.asyncSelect.main,
    padding: '10px'
})

export const OutputContainer = styled(Box)({
    background: theme.palette.asyncSelect.main,
    padding: '10px',
    marginBottom: '10px'
})

export const AdditionalParamsButton = styled(Button)({
    borderRadius: '25px'
    // other styles as necessary
})

