import PropTypes from 'prop-types'
import { useState } from 'react'
import styled from '@mui/material/styles/styled'
import { Box, Typography, Divider, Button } from '@mui/material'

import MainCard from 'ui-component/cards/MainCard'
import NodeInputHandler from 'views/canvas/NodeInputHandler'
import NodeOutputHandler from 'views/canvas/NodeOutputHandler'
import AdditionalParamsDialog from 'ui-component/dialog/AdditionalParamsDialog'

import { baseURL } from 'store/constant'

const CardWrapper = styled(MainCard)(({ theme }) => ({
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

// Extracted reusable components
const AvatarWrapper = styled('div')(({ theme }) => ({
  ...theme.typography.commonAvatar,
  ...theme.typography.largeAvatar,
  borderRadius: '50%',
  backgroundColor: 'white',
  cursor: 'grab'
}))

const AvatarImage = styled('img')({
  width: '100%',
  height: '100%',
  padding: 5,
  objectFit: 'contain'
})

const InputsWrapper = styled(Box)(({ theme }) => ({
  background: theme.palette.asyncSelect.main,
  p: 1
}))

// Refactored component
function MarketplaceCanvasNode({ data }) {
  const { palette: { primary, text, asyncSelect }, typography } = useTheme()

  const [showDialog, setShowDialog] = useState(false)
  const [dialogProps, setDialogProps] = useState({})

  const handleDialogClicked = () => {
    const filteredParams = data.inputParams.filter(param => param.additionalParams)
    const dialogProps = {
      data,
      inputParams: filteredParams,
      disabled: true,
      confirmButtonName: 'Save',
      cancelButtonName: 'Cancel'
    }
    setDialogProps(dialogProps)
    setShowDialog(true)
  }

  const hasAdditionalParams = !!data.inputParams.find(param => param.additionalParams)

  return (
    <>
      <CardWrapper
        content={false}
        sx={{
          padding: 0,
          borderColor: data.selected ? primary.main : text.secondary
        }}
        border={false}
      >
        <Box>
          <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
            <Box style={{ width: 50, marginRight: 10, padding: 5 }}>
              <AvatarWrapper>
                <AvatarImage src={`${baseURL}/api/v1/node-icon/${data.name}`} alt='Notification' />
              </AvatarWrapper>
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
          {(data.inputAnchors.length > 0 || data.inputParams.length > 0) && (
            <>
              <Divider />
              <InputsWrapper>
                <Typography
                  sx={{
                    fontWeight: 500,
                    textAlign: 'center'
                  }}
                >
                  Inputs
                </Typography>
              </InputsWrapper>
              <Divider />
            </>
          )}
          {data.inputAnchors.map((inputAnchor, index) => (
            <NodeInputHandler disabled key={index} inputAnchor={inputAnchor} data={data} />
          ))}
          {data.inputParams.map((inputParam, index) => (
            <NodeInputHandler disabled key={index} inputParam={inputParam} data={data} />
          ))}
          {hasAdditionalParams && (
            <div style={{ textAlign: 'center' }}>
              <Button
                sx={{ borderRadius: 25, width: '90%', mb: 2 }}
                variant='outlined'
                onClick={handleDialogClicked}
              >
                Additional Parameters
              </Button>
            </div>
          )}
          <Divider />
          <InputsWrapper>
            <Typography
              sx={{
                fontWeight: 500,
                textAlign: 'center'
              }}
            >
              Output
            </Typography>
          </InputsWrapper>
          <Divider />
          {data.outputAnchors.map((outputAnchor, index) => (
            <NodeOutputHandler disabled key={index} outputAnchor={outputAnchor} data={data} />
          ))}
        </Box>
      </CardWrapper>
      <AdditionalParamsDialog show={showDialog} dialogProps={dialogProps} onCancel={() => setShowDialog(false)} />
    </>
  )
}

MarketplaceCanvasNode.propTypes = {
  data: PropTypes.object
}

export default MarketplaceCanvasNode