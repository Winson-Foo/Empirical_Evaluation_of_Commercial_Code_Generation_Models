// AdditionalParamsDialog.js
import { createPortal } from 'react-dom';
import { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { Dialog } from '@mui/material';
import PerfectScrollbar from 'react-perfect-scrollbar';
import NodeInputHandler from 'views/canvas/NodeInputHandler';
import Content from './DialogContent';

function AdditionalParamsDialog(props) {
  const portalElement = document.getElementById('portal');
  const { show, dialogProps, onCancel } = props;

  const [inputParams, setInputParams] = useState([]);
  const [data, setData] = useState({});

  useEffect(() => {
    if (dialogProps.inputParams) setInputParams(dialogProps.inputParams);
    if (dialogProps.data) setData(dialogProps.data);
    return () => {
      setInputParams([]);
      setData({});
    };
  }, [dialogProps]);

  const component = show ? (
    <Dialog
      onClose={onCancel}
      open={show}
      fullWidth
      maxWidth='sm'
      aria-labelledby='alert-dialog-title'
      aria-describedby='alert-dialog-description'
    >
      <Content inputParams={inputParams} dialogProps={dialogProps} data={data} />
    </Dialog>
  ) : null;

  return createPortal(component, portalElement);
}

AdditionalParamsDialog.propTypes = {
  show: PropTypes.bool.isRequired,
  dialogProps: PropTypes.object.isRequired,
  onCancel: PropTypes.func.isRequired,
};

export default AdditionalParamsDialog;

// DialogContent.js
import { DialogContent } from '@mui/material';
import PropTypes from 'prop-types';
import PerfectScrollbar from 'react-perfect-scrollbar';
import NodeInputHandler from 'views/canvas/NodeInputHandler';

function Content(props) {
  const { inputParams, dialogProps, data } = props;
  return (
    <DialogContent>
      <PerfectScrollbar
        style={{
          height: '100%',
          maxHeight: 'calc(100vh - 220px)',
          overflowX: 'hidden',
        }}
      >
        {inputParams.map((inputParam, index) => (
          <NodeInputHandler
            disabled={dialogProps.disabled}
            key={index}
            inputParam={inputParam}
            data={data}
            isAdditionalParams={true}
          />
        ))}
      </PerfectScrollbar>
    </DialogContent>
  );
}

Content.propTypes = {
  inputParams: PropTypes.array.isRequired,
  dialogProps: PropTypes.object.isRequired,
  data: PropTypes.object.isRequired,
};

export default Content;

// AdditionalParamsDialog.defaultProps
AdditionalParamsDialog.defaultProps = {
  show: false,
};

