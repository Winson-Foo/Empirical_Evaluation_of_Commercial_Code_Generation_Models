// ==============================|| LOGO ||============================== //

import { useSelector } from 'react-redux'
import logoConfig from '../../config/logoConfig'

const Logo = ({ height, width }) => {
    const customization = useSelector((state) => state.customization)

    return (
        <div style={{ alignItems: 'center', display: 'flex', flexDirection: 'row' }}>
            <img
                style={{ objectFit: 'contain', height: height, width: width }}
                src={customization.isDarkMode ? logoConfig.dark : logoConfig.light}
                alt='Flowise'
            />
        </div>
    )
}

export default Logo