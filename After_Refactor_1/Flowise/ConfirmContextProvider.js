import { useReducer } from 'react';
import PropTypes from 'prop-types';
import ConfirmContext from './ConfirmContext';
import confirmReducer, { confirmInitialState } from '../reducers/dialogReducer';

/* 
    This component provides the ConfirmContext to its children. 
    It uses the useReducer hook to manage confirm dialogs.
*/
const ConfirmContextProvider = ({ children }) => {
    const [state, dispatch] = useReducer(confirmReducer, confirmInitialState);

    return <ConfirmContext.Provider value={[state, dispatch]}>{children}</ConfirmContext.Provider>;
};

ConfirmContextProvider.propTypes = {
    children: PropTypes.any,
};

export default ConfirmContextProvider;

