import PropTypes from 'prop-types'
import NodeInputHandler from 'views/canvas/NodeInputHandler'

const InputField = ({ disabled, inputParam, data }) => {
  return (
    <NodeInputHandler
      disabled={disabled}
      inputParam={inputParam}
      data={data}
      isAdditionalParams={true}
    />
  )
}

InputField.propTypes = {
  disabled: PropTypes.bool,
  inputParam: PropTypes.object,
  data: PropTypes.object,
}

export default InputField