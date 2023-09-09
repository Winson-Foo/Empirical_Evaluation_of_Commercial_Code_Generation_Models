// CanvasNode.js

import PropTypes from 'prop-types';
import { useContext, useState } from 'react';
import { styled, useTheme } from '@mui/material/styles';
import { IconButton, Box, Typography, Divider, Button } from '@mui/material';
import { IconTrash, IconCopy } from '@tabler/icons';
import MainCard from 'ui-component/cards/MainCard';
import NodeInputHandler from './NodeInputHandler';
import NodeOutputHandler from './NodeOutputHandler';
import AdditionalParamsDialog from 'ui-component/dialog/AdditionalParamsDialog';
import { flowContext } from 'store/context/ReactFlowContext';

import { baseURL } from 'store/constant';

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
    borderColor: theme.palette.primary.main,
  },
}));

const NodeIconWrapper = styled(Box)(({ theme }) => ({
  width: 50,
  marginRight: 10,
  padding: 5,
  borderRadius: '50%',
  backgroundColor: 'white',
  cursor: 'grab',
  ...theme.typography.commonAvatar,
  ...theme.typography.largeAvatar,
}));

function NodeIcon({ name }) {
  const theme = useTheme();

  return (
    <NodeIconWrapper>
      <img
        style={{ width: '100%', height: '100%', padding: 5, objectFit: 'contain' }}
        src={`${baseURL}/api/v1/node-icon/${name}`}
        alt='Notification'
      />
    </NodeIconWrapper>
  );
}

NodeIcon.propTypes = {
  name: PropTypes.string.isRequired,
};

function NodeDetails({ data }) {
  const { deleteNode, duplicateNode } = useContext(flowContext);
  const theme = useTheme();

  return (
    <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
      <NodeIcon name={data.name} />
      <Box>
        <Typography sx={{ fontSize: '1rem', fontWeight: 500 }}>{data.label}</Typography>
      </Box>
      <div style={{ flexGrow: 1 }}></div>
      <IconButton
        title='Duplicate'
        onClick={() => {
          duplicateNode(data.id);
        }}
        sx={{ height: 35, width: 35, '&:hover': { color: theme?.palette.primary.main } }}
        color={theme?.customization?.isDarkMode ? theme.colors?.paper : 'inherit'}
      >
        <IconCopy />
      </IconButton>
      <IconButton
        title='Delete'
        onClick={() => {
          deleteNode(data.id);
        }}
        sx={{ height: 35, width: 35, mr: 1, '&:hover': { color: 'red' } }}
        color={theme?.customization?.isDarkMode ? theme.colors?.paper : 'inherit'}
      >
        <IconTrash />
      </IconButton>
    </div>
  );
}

NodeDetails.propTypes = {
  data: PropTypes.object.isRequired,
};

function NodeInputs({ inputs, onAdditionalParamsClick }) {
  return (
    <>
      {inputs.length > 0 && (
        <>
          <Divider />
          <Box sx={{ background: theme.palette.asyncSelect.main, p: 1 }}>
            <Typography sx={{ fontWeight: 500, textAlign: 'center' }}>Inputs</Typography>
          </Box>
          <Divider />
        </>
      )}
      {inputs.map((input, index) => (
        <NodeInputHandler key={index} input={input} />
      ))}
      {inputs.find((param) => param.additionalParams) && (
        <Box sx={{ textAlign: 'center' }}>
          <Button sx={{ borderRadius: 25, width: '90%', mb: 2 }} variant='outlined' onClick={onAdditionalParamsClick}>
            Additional Parameters
          </Button>
        </Box>
      )}
    </>
  );
}

NodeInputs.propTypes = {
  inputs: PropTypes.array.isRequired,
  onAdditionalParamsClick: PropTypes.func.isRequired,
};

function NodeOutputs({ outputs }) {
  return (
    <>
      <Divider />
      <Box sx={{ background: theme.palette.asyncSelect.main, p: 1 }}>
        <Typography sx={{ fontWeight: 500, textAlign: 'center' }}>Outputs</Typography>
      </Box>
      <Divider />
      {outputs.map((output, index) => (
        <NodeOutputHandler key={index} output={output} />
      ))}
    </>
  );
}

NodeOutputs.propTypes = {
  outputs: PropTypes.array.isRequired,
};

function CanvasNode({ data }) {
  const theme = useTheme();
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [dialogProps, setDialogProps] = useState({});

  function handleAdditionalParamsClick() {
    const additionalParams = data.inputParams.filter((param) => param.additionalParams);

    if (additionalParams.length > 0) {
      setIsDialogOpen(true);
      setDialogProps({
        data,
        inputParams: additionalParams,
        confirmButtonName: 'Save',
        cancelButtonName: 'Cancel',
      });
    }
  }

  return (
    <>
      <CardWrapper content={false} sx={{ padding: 0, borderColor: data.selected ? theme.palette.primary.main : theme.palette.text.secondary }} border={false}>
        <Box>
          <NodeDetails data={data} />
          <NodeInputs inputs={data.inputAnchors} onAdditionalParamsClick={handleAdditionalParamsClick} />
          <NodeInputs inputs={data.inputParams} onAdditionalParamsClick={handleAdditionalParamsClick} />
          <NodeOutputs outputs={data.outputAnchors} />
        </Box>
      </CardWrapper>
      <AdditionalParamsDialog show={isDialogOpen} dialogProps={dialogProps} onCancel={() => setIsDialogOpen(false)} />
    </>
  );
}

CanvasNode.propTypes = {
  data: PropTypes.object.isRequired,
};

export default CanvasNode;

