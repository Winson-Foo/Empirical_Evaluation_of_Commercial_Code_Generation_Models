import React from 'react';
import PropTypes from 'prop-types';
import Editor from 'react-simple-code-editor';
import { highlight, languages } from 'prismjs/components/prism-core';
import { Prism as PrismCLike } from 'prismjs/components/prism-clike';
import { Prism as PrismJS } from 'prismjs/components/prism-javascript';
import { Prism as PrismJSON } from 'prismjs/components/prism-json';
import { Prism as PrismMarkup } from 'prismjs/components/prism-markup';
import { useTheme } from '@mui/material/styles';
import './prism-dark.css';

const CodeEditor = ({ value, placeholder, disabled, type, style, onValueChange, onMouseUp, onBlur }) => {
  const theme = useTheme();

  const getLanguage = () => {
    switch (type) {
      case 'json':
        return languages.json;
      case 'markup':
        return languages.markup;
      case 'clike':
      default:
        return languages.clike;
    }
  };

  return (
    <Editor
      disabled={disabled}
      value={value}
      placeholder={placeholder}
      highlight={(code) => highlight(code, getLanguage())}
      padding={10}
      onValueChange={onValueChange}
      onMouseUp={onMouseUp}
      onBlur={onBlur}
      style={{
        ...style,
        background: theme.palette.codeEditor.main,
      }}
      textareaClassName='editor__textarea'
    />
  );
};

CodeEditor.propTypes = {
  value: PropTypes.string,
  placeholder: PropTypes.string,
  disabled: PropTypes.bool,
  type: PropTypes.oneOf(['json', 'markup', 'clike']),
  style: PropTypes.object,
  onValueChange: PropTypes.func,
  onMouseUp: PropTypes.func,
  onBlur: PropTypes.func,
};

CodeEditor.defaultProps = {
  disabled: false,
  type: 'clike',
};

export default CodeEditor;