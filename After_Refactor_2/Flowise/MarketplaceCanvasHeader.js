import PropTypes from 'prop-types';
import { useNavigate } from 'react-router-dom';
import { useTheme } from '@mui/material/styles';
import { Avatar, ButtonBase, Typography, Stack } from '@mui/material';
import { StyledButton } from 'ui-component/button/StyledButton';
import { IconCopy, IconChevronLeft } from '@tabler/icons';

const MarketplaceCanvasHeader = ({ flowName, flowData, onChatflowCopy }) => {
  const theme = useTheme();
  const navigate = useNavigate();

  const handleCopy = () => {
    onChatflowCopy(flowData);
  };

  return (
    <>
      <ButtonBase title="Back" sx={{ borderRadius: '50%' }}>
        <Avatar
          variant="rounded"
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
          color="inherit"
          onClick={() => navigate(-1)}
        >
          <IconChevronLeft stroke={1.5} size="1.3rem" />
        </Avatar>
      </ButtonBase>
      <Stack flexGrow={1} flexDirection="row">
        <Typography
          sx={{
            fontSize: '1.5rem',
            fontWeight: 600,
            ml: 2,
          }}
        >
          {flowName}
        </Typography>
      </Stack>
      <StyledButton
        color="secondary"
        variant="contained"
        title="Use Chatflow Template"
        onClick={handleCopy}
        startIcon={<IconCopy />}
        sx={{ ml: 1 }}
      >
        Use Chatflow Template
      </StyledButton>
    </>
  );
};

MarketplaceCanvasHeader.propTypes = {
  flowName: PropTypes.string,
  flowData: PropTypes.object,
  onChatflowCopy: PropTypes.func,
};

export default MarketplaceCanvasHeader;

