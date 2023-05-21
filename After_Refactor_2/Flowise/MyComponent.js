// MyComponent.js

import { connect } from 'react-redux';
import { setDirty, removeDirty, setChatFlow } from '../actions';

const MyComponent = ({ isDirty, chatflow, setDirty, removeDirty, setChatFlow }) => {
  // component logic here
}

const mapStateToProps = state => ({
  isDirty: state.canvasReducer.isDirty,
  chatflow: state.canvasReducer.chatflow,
});

const mapDispatchToProps = {
  setDirty,
  removeDirty,
  setChatFlow,
};

export default connect(mapStateToProps, mapDispatchToProps)(MyComponent);