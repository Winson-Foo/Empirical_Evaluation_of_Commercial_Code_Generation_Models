import { createPortal } from 'react-dom';
import { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { Dialog, DialogContent } from '@mui/material';
import PerfectScrollbar from 'react-perfect-scrollbar';
import NodeInputHandler from 'views/canvas/NodeInputHandler';

const AdditionalParamsDialog = ({ show, dialogProps, onCancel }) => {
  const portalElement = document.getElementById('portal');

  const [inputParams, setInputParams] = useState([]);
  const [data, setData] = useState({});

  useEffect(() => {
    setInputParams(dialogProps.inputParams || []);
    setData(dialogProps.data || {});

    return () => {
      setInputParams([]);
      setData({});
    };
  }, [dialogProps]);

  const renderInputHandlers = () => {
    return inputParams.map((inputParam, index) => (
      <NodeInputHandler
        key={index}
        disabled={dialogProps.disabled}
        inputParam={inputParam}
        data={data}
        isAdditionalParams
      />
    ));
  };

  const renderDialogContent = () => {
    return (
      <PerfectScrollbar
        style={{
          height: '100%',
          maxHeight: 'calc(100vh - 220px)',
          overflowX: 'hidden'
        }}
      >
        {renderInputHandlers()}
      </PerfectScrollbar>
    );
  };

  const renderDialog = () => {
    return (
      <Dialog
        onClose={onCancel}
        open={show}
        fullWidth
        maxWidth="sm"
        aria-labelledby="alert-dialog-title"
        aria-describedby="alert-dialog-description"
      >
        <DialogContent>
          {renderDialogContent()}
        </DialogContent>
      </Dialog>
    );
  };

  return createPortal(renderDialog(), portalElement);
};

AdditionalParamsDialog.propTypes = {
  show: PropTypes.bool,
  dialogProps: PropTypes.object,
  onCancel: PropTypes.func
};

export default AdditionalParamsDialog;