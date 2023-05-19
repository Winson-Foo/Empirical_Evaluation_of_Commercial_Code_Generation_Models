import React from 'react';
import PropTypes from 'prop-types';
import Editor from 'react-simple-code-editor';
import { highlight, languages } from 'prismjs/components/prism-core';
import 'prismjs/components/prism-clike';
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-markup';
import './prism-light.css';
import { useTheme } from '@mui/material/styles';

const CODE_TYPES = {
  json: languages.json,
  js: languages.js
};

const LightCodeEditor = ({
  value = '',
  placeholder = '',
  disabled = false,
  type = 'js',
  style = {},
  onChange = () => {},
  onMouseUp = () => {},
  onBlur = () => {}
}) => {
  const theme = useTheme();

  return (
    <Editor
      value={value}
      placeholder={placeholder}
      highlight={code => highlight(code, CODE_TYPES[type])}
      padding={10}
      onValueChange={onChange}
      onMouseUp={onMouseUp}
      onBlur={onBlur}
      disabled={disabled}
      style={{ ...style, background: theme.palette.card.main }}
      textareaClassName="editor-textarea"
    />
  );
};

LightCodeEditor.propTypes = {
  value: PropTypes.string,
  placeholder: PropTypes.string,
  disabled: PropTypes.bool,
  type: PropTypes.string,
  style: PropTypes.object,
  onChange: PropTypes.func,
  onMouseUp: PropTypes.func,
  onBlur: PropTypes.func
};

export default LightCodeEditor;

