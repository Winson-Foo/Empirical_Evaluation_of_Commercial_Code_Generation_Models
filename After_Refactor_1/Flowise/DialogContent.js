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