import PropTypes from 'prop-types'
import { forwardRef } from 'react'

// material-ui
import { useTheme } from '@mui/material/styles'
import { Card, CardContent, CardHeader, Divider, Typography } from '@mui/material'

// constant
const headerSX = {
    '& .MuiCardHeader-action': {
        margin-right: 0;
    }
}

// ==============================|| CUSTOM MAIN CARD ||============================== //

const MainCard = forwardRef(function MainCard(props, ref) {
    const theme = useTheme()

    const {
        border = true,
        boxShadow = true,
        children,
        content = true,
        contentClass = '',
        contentSX = {},
        darkTitle = false,
        secondary,
        shadow = '0 2px 14px 0 rgb(32 40 45 / 8%)',
        sx = {},
        title,
        ...others
    } = props;

    /**
     * Render card header and action.
     *
     * @returns {JSX.Element}
     */
    const renderCardHeader = () => {
        if (!title) return null;

        return (
            <CardHeader
                sx={headerSX}
                title={
                    darkTitle ?
                        <Typography variant='h3'>{title}</Typography> :
                        title
                }
                action={secondary}
            />
        );
    };

    /**
     * Render card content.
     *
     * @returns {JSX.Element}
     */
    const renderCardContent = () => {
        if (!children || !content) return null;

        return (
            <CardContent sx={contentSX} className={contentClass}>
                {children}
            </CardContent>
        );
    };

    return (
        <Card
            ref={ref}
            {...others}
            sx={{
                border: border ? '1px solid' : 'none',
                borderColor: theme.palette.primary[200] + 75,
                ':hover': {
                    boxShadow: boxShadow ? shadow : 'inherit'
                },
                ...sx
            }}
        >
            {/* card header and action */}
            {renderCardHeader()}

            {/* content & header divider */}
            {title && <Divider />}

            {/* card content */}
            {renderCardContent()}
        </Card>
    )
});

MainCard.propTypes = {
    border: PropTypes.bool,
    boxShadow: PropTypes.bool,
    children: PropTypes.node,
    content: PropTypes.bool,
    contentClass: PropTypes.string,
    contentSX: PropTypes.object,
    darkTitle: PropTypes.bool,
    secondary: PropTypes.oneOfType([PropTypes.node, PropTypes.string, PropTypes.object]),
    shadow: PropTypes.string,
    sx: PropTypes.object,
    title: PropTypes.oneOfType([PropTypes.node, PropTypes.string, PropTypes.object])
};

MainCard.defaultProps = {
    boxShadow: true,
    darkTitle: false,
};

export default MainCard;