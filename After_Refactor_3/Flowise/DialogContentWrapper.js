import PerfectScrollbar from 'react-perfect-scrollbar'
import PropTypes from 'prop-types'
import InputField from './InputField'

const DialogContentWrapper = ({ inputParams, data, disabled }) => {
  return (
    <PerfectScrollbar
      style={{
        height: '100%',
        maxHeight: 'calc(100vh - 220px)',
        overflowX: 'hidden',
      }}
    >
      {inputParams.map((inputParam, index) => (
        <InputField
          disabled={disabled}
          key={index}
          inputParam={inputParam}
          data={data}
        />
      ))}
    </PerfectScrollbar>
  )
}

DialogContentWrapper.propTypes = {
  inputParams: PropTypes.array,
  data: PropTypes.object,
  disabled: PropTypes.bool,
}

export default DialogContentWrapper