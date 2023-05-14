import { useState } from 'react';
import PropTypes from 'prop-types';
import { FormControl } from '@mui/material';
import ReactJson from 'react-json-view';

const JsonEditorInput = ({ value, onChange, disabled = false, isDarkMode = false }) => {
  
  // Parse incoming JSON into a state hook for rendering
  const [jsonValue, setJsonValue] = useState(value ? JSON.parse(value) : {});

  // Format JSON for copying to clipboard
  const onClipboardCopy = (e) => {
    const source = e.source;
    const formattedJson = Array.isArray(source) || typeof source === 'object'
      ? JSON.stringify(source, null, '  ')
      : source;
    navigator.clipboard.writeText(formattedJson);
  };

  return (
    <FormControl sx={{ mt: 1, width: '100%' }} size='small'>
      {/* Render JSON viewer and editor */}
      <ReactJson
        theme={isDarkMode ? 'ocean' : 'rjv-default'}
        style={{ padding: 10, borderRadius: 10 }}
        src={jsonValue}
        name={null}
        enableClipboard={(e) => onClipboardCopy(e)}
        quotesOnKeys={false}
        displayDataTypes={false}
        onEdit={(edit) => {
          setJsonValue(edit.updated_src);
          onChange(JSON.stringify(edit.updated_src));
        }}
        onAdd={() => {
          // console.log(add);
        }}
        onDelete={(deleteobj) => {
          setJsonValue(deleteobj.updated_src);
          onChange(JSON.stringify(deleteobj.updated_src));
        }}
        {...(!disabled && {
          onEdit: (edit) => {
            setJsonValue(edit.updated_src);
            onChange(JSON.stringify(edit.updated_src));
          },
          onAdd: () => {
            // console.log(add);
          },
          onDelete: (deleteobj) => {
            setJsonValue(deleteobj.updated_src);
            onChange(JSON.stringify(deleteobj.updated_src));
          },
        })}
      />
    </FormControl>
  );
}

JsonEditorInput.propTypes = {
  value: PropTypes.string,
  onChange: PropTypes.func.isRequired,
  disabled: PropTypes.bool,
  isDarkMode: PropTypes.bool,
};

JsonEditorInput.defaultProps = {
  disabled: false,
  isDarkMode: false,
};

export default JsonEditorInput;

