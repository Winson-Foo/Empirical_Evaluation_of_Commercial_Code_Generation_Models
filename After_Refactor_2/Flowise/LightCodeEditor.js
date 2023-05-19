// Imports
import React from 'react';
import Editor from 'react-simple-code-editor';
import { highlight, languages } from 'prismjs/components/prism-core';
import 'prismjs/components/prism-clike';
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-markup';
import './prism-light.css';
import { useTheme } from '@mui/material/styles';

// Named Export
export const LightCodeEditor: React.FC<LightCodeEditorProps> = ({ 
  value, 
  placeholder, 
  disabled = false, 
  type, 
  style, 
  onValueChange, 
  onMouseUp, 
  onBlur 
}) => {
  const theme = useTheme();

  return (
    <Editor
      disabled={disabled}
      value={value}
      placeholder={placeholder}
      highlight={(code) => highlight(code, type === 'json' ? languages.json : languages.js)}
      padding={10}
      onValueChange={onValueChange}
      onMouseUp={onMouseUp}
      onBlur={onBlur}
      style={{
        ...style,
        background: theme.palette.card.main
      }}
      textareaClassName='editor__textarea'
    />
  );
};

// The LightCodeEditorProps interface
interface LightCodeEditorProps {
  value: string;
  placeholder?: string;
  disabled?: boolean;
  type?: 'json' | 'js';
  style?: React.CSSProperties;
  onValueChange?: (code: string) => void;
  onMouseUp?: () => void;
  onBlur?: () => void;
}

// Export the LightCodeEditorProps interface
export type { LightCodeEditorProps };

