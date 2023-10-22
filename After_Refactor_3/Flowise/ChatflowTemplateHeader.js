// ChatflowTemplateHeader.js

import PropTypes from 'prop-types';
import { useNavigate } from 'react-router-dom';
import { useTheme } from '@mui/material/styles';
import { Box, ButtonBase, Stack } from '@mui/material';
import { IconCopy, IconChevronLeft } from '@tabler/icons';
import { StyledButton } from 'ui-component/button/StyledButton';
import Avatar from './Avatar';
import Typography from './Typography';

const ChatflowTemplateHeader = ({
  flowName = '',
  flowData = {},
  onChatflowCopy = () => {},
}) => {
  const theme = useTheme();
  const navigate = useNavigate();

  return (
    <>
      <Box>
        <ButtonBase title='Back' sx={{ borderRadius: '50%' }}>
          <Avatar
            variant='rounded'
            sx={{
              ...theme.typography.commonAvatar,
              ...theme.typography.mediumAvatar,
              transition: 'all .2s ease-in-out',
              background: theme.palette.secondary.light,
              color: theme.palette.secondary.dark,
              '&:hover': {
                background: theme.palette.secondary.dark,
                color: theme.palette.secondary.light,
              },
            }}
            color='inherit'
            onClick={() => navigate(-1)}
          >
            <IconChevronLeft stroke={1.5} size='1.3rem' />
          </Avatar>
        </ButtonBase>
      </Box>
      <Box sx={{ flexGrow: 1 }}>
        <Stack flexDirection='row'>
          <Typography
            variant='h5'
            sx={{
              fontWeight: 600,
              ml: 2,
            }}
          >
            {flowName}
          </Typography>
        </Stack>
      </Box>
      <Box>
        <StyledButton
          color='secondary'
          variant='contained'
          title='Use Chatflow'
          onClick={() => onChatflowCopy(flowData)}
          startIcon={<IconCopy />}
        >
          Use Template
        </StyledButton>
      </Box>
    </>
  );
};

ChatflowTemplateHeader.propTypes = {
  flowName: PropTypes.string,
  flowData: PropTypes.object,
  onChatflowCopy: PropTypes.func,
};

export default ChatflowTemplateHeader;