import { createPortal } from 'react-dom';
import { useState, useEffect } from 'react';
import { useDispatch } from 'react-redux';
import PropTypes from 'prop-types';
import {
  Box,
  Typography,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Stack,
  IconButton,
  OutlinedInput,
  Popover
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { IconX, IconCopy } from '@tabler/icons';
import useNotifier from 'utils/useNotifier';
import apikeyApi from 'api/apikey';
import { StyledButton } from 'ui-component/button/StyledButton';

function ApiKeyForm({ type, key, keyName, setKeyName }) {
  const theme = useTheme();
  
  const [anchorEl, setAnchorEl] = useState(null);
  const openPopOver = Boolean(anchorEl);
  
  const handleClosePopOver = () => {
    setAnchorEl(null);
  };
  
  const handleCopyToClipboard = () => {
    navigator.clipboard.writeText(key.apiKey)
    setAnchorEl(event.currentTarget);
    setTimeout(() => {
        handleClosePopOver();
    }, 1500)
  }

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant='overline'>Key Name</Typography>
      <OutlinedInput
        id='keyName'
        type='string'
        fullWidth
        placeholder='My New Key'
        value={keyName}
        name='keyName'
        onChange={(e) => setKeyName(e.target.value)}
      />
      {type === 'EDIT' && (
        <Box sx={{ p: 2 }}>
          <Typography variant='overline'>API Key</Typography>
          <Stack direction='row' sx={{ mb: 1 }}>
            <Typography
                sx={{
                    p: 1,
                    borderRadius: 10,
                    backgroundColor: theme.palette.primary.light,
                    width: 'max-content',
                    height: 'max-content'
                }}
                variant='h5'
            >
              {key.apiKey}
            </Typography>
            <IconButton
                title='Copy API Key'
                color='success'
                onClick={handleCopyToClipboard}
            >
              <IconCopy />
            </IconButton>
            <Popover
              open={openPopOver}
              anchorEl={anchorEl}
              onClose={handleClosePopOver}
              anchorOrigin={{
                vertical: 'top',
                horizontal: 'right'
              }}
              transformOrigin={{
                  vertical: 'top',
                  horizontal: 'left'
              }}
            >
              <Typography variant='h6' sx={{ pl: 1, pr: 1, color: 'white', background: theme.palette.success.dark }}>
                Copied!
              </Typography>
            </Popover>
          </Stack>
        </Box>
      )}
    </Box>
  );
}

function ApiKeyFormDialog({ show, dialogProps, onCancel, onConfirm }) {
  const dispatch = useDispatch();

  useNotifier();
  
  const enqueueSnackbar = (...args) => dispatch(enqueueSnackbarAction(...args));
  const closeSnackbar = (...args) => dispatch(closeSnackbarAction(...args));
  
  const [keyName, setKeyName] = useState('');
  
  useEffect(() => {
    if (dialogProps.type === 'EDIT' && dialogProps.key) {
        setKeyName(dialogProps.key.keyName);
    } else if (dialogProps.type === 'ADD') {
        setKeyName('');
    }
  }, [dialogProps])
  
  const handleSubmit = async () => {
    try {
        const data = {
            keyName: keyName
        }
        let response;
        if (dialogProps.type === 'ADD') {
            response = await apikeyApi.createNewAPI(data);
        } else if (dialogProps.type === 'EDIT' && dialogProps.key) {
            response = await apikeyApi.updateAPI(dialogProps.key.id, data);
        }
        if (response.data) {
            const successMessage = 
                dialogProps.type === 'ADD' ? 'New API key added' : 'API Key saved';
            enqueueSnackbar({
              message: successMessage,
              options: {
                  key: new Date().getTime() + Math.random(),
                  variant: 'success',
                  action: (key) => (
                      <Button style={{ color: 'white' }} onClick={() => closeSnackbar(key)}>
                          <IconX />
                      </Button>
                  )
              }
            });
            onConfirm();
        }
    } catch (error) {
        const errorData = error.response.data || `${error.response.status}: ${error.response.statusText}`;
        const errorMessage = 
            dialogProps.type === 'ADD' ? `Failed to add new API key: ${errorData}` : `Failed to save API key: ${errorData}`;
        enqueueSnackbar({
          message: errorMessage,
          options: {
              key: new Date().getTime() + Math.random(),
              variant: 'error',
              persist: true,
              action: (key) => (
                  <Button style={{ color: 'white' }} onClick={() => closeSnackbar(key)}>
                      <IconX />
                  </Button>
              )
          }
        });
        onCancel();
    }
  }

  const portalElement = document.getElementById('portal');
  
  return createPortal(
    <Dialog
        fullWidth
        maxWidth='sm'
        open={show}
        onClose={onCancel}
        aria-labelledby='alert-dialog-title'
        aria-describedby='alert-dialog-description'
    >
        <DialogTitle sx={{ fontSize: '1rem' }} id='alert-dialog-title'>
            {dialogProps.title}
        </DialogTitle>
        <DialogContent>
            <ApiKeyForm 
              type={dialogProps.type} 
              key={dialogProps.key}
              keyName={keyName}
              setKeyName={setKeyName}
            />
        </DialogContent>
        <DialogActions>
            <StyledButton 
              variant='contained' 
              onClick={handleSubmit}
            >
                {dialogProps.confirmButtonName}
            </StyledButton>
        </DialogActions>
    </Dialog>,
    portalElement
  );
}

ApiKeyFormDialog.propTypes = {
    show: PropTypes.bool,
    dialogProps: PropTypes.object,
    onCancel: PropTypes.func,
    onConfirm: PropTypes.func
}

export default ApiKeyFormDialog;