// components/DarkCodeEditor.js
import Editor from 'react-simple-code-editor';
import PropTypes from 'prop-types';
import { highlight, languages } from 'prismjs/components/prism-core';
import 'prismjs/components/prism-clike';
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-markup';
import './prism-dark.css';
import { useTheme } from '@mui/material/styles';

const DarkCodeEditor = ({
  value,
  placeholder,
  disabled = false,
  type,
  style,
  onValueChange,
  onMouseUp,
  onBlur,
}) => {
  const theme = useTheme();

  return (
    <Editor
      disabled={disabled}
      value={value}
      placeholder={placeholder}
      highlight={(code) =>
        highlight(
          code,
          type === 'json' ? languages.json : languages.js
        )
      }
      padding={10}
      onValueChange={onValueChange}
      onMouseUp={onMouseUp}
      onBlur={onBlur}
      style={{
        ...style,
        background: theme.palette.codeEditor.main,
      }}
      textareaClassName="editor__textarea"
    />
  );
};

DarkCodeEditor.propTypes = {
  value: PropTypes.string,
  placeholder: PropTypes.string,
  disabled: PropTypes.bool,
  type: PropTypes.string,
  style: PropTypes.object,
  onValueChange: PropTypes.func,
  onMouseUp: PropTypes.func,
  onBlur: PropTypes.func,
};

export default DarkCodeEditor;


// app.js
import DarkCodeEditor from './components/DarkCodeEditor';

const App = () => {
  // example usage of DarkCodeEditor
  return (
    <div>
      <h1>My Code Editor</h1>
      <DarkCodeEditor
        value=""
        placeholder="Enter your code here..."
        type="javascript"
        onValueChange={(code) => console.log(code)}
      />
    </div>
  );
};

