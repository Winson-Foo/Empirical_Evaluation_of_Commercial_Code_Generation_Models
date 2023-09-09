import PropTypes from 'prop-types';
import { useNavigate } from 'react-router-dom';

import { useTheme, styled } from '@mui/material/styles';
import { Avatar, Box, ButtonBase, Typography, Stack } from '@mui/material';
import { IconCopy, IconChevronLeft } from '@tabler/icons';

import { StyledButton } from 'ui-component/button/StyledButton';

// ==============================|| CANVAS HEADER ||============================== //

const MarketCanvasHeader = styled(Box)(({theme}) => ({
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    margin: '2rem 0',
    '& .back': {
        borderRadius: '50%',
        '&:hover': {
            background: theme.palette.secondary.dark,
            color: theme.palette.secondary.light
        }
    },
    '& .title': {
        flexGrow: 1,
        display: 'flex',
        alignItems: 'center'
    },
    '& .button': {

    }
}))

const BackButton = ({navigate}) => {
    return (
        <ButtonBase title='Back' className='back'>
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
                        color: theme.palette.secondary.light
                    }
                }}
                color='inherit'
                onClick={() => navigate(-1)}
            >
                <IconChevronLeft stroke={1.5} size='1.3rem' />
            </Avatar>
        </ButtonBase>
    )
}

const Title = ({title}) => {
    return (
        <Box className='title'>
            <Stack flexDirection='row'>
                <Typography
                    sx={{
                        fontSize: '1.5rem',
                        fontWeight: 600,
                        ml: 2
                    }}
                >
                    {title}
                </Typography>
            </Stack>
        </Box>
    )
}

const ChatflowButton = ({onChatflowCopy, flowData}) => {
    return (
        <Box className='button'>
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
    )
}

const MarketplaceCanvasHeader = ({ flowName, flowData, onChatflowCopy }) => {
    const theme = useTheme()
    const navigate = useNavigate()

    return (
        <MarketCanvasHeader>
            <BackButton navigate={navigate}/>
            <Title title={flowName}/>
            <ChatflowButton onChatflowCopy={onChatflowCopy} flowData={flowData}/>
        </MarketCanvasHeader>
    )
}

MarketplaceCanvasHeader.propTypes = {
    flowName: PropTypes.string,
    flowData: PropTypes.object,
    onChatflowCopy: PropTypes.func
}

export default MarketplaceCanvasHeader