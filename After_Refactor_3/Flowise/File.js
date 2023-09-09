import { useState } from 'react';
import PropTypes from 'prop-types';
import { useTheme } from '@mui/material/styles';
import { FormControl, Button } from '@mui/material';
import { IconUpload } from '@tabler/icons';
import { getFileName } from 'utils/genericHelper';

const File = ({ value, fileType, onChange, disabled = false }) => {
  const theme = useTheme();
  const [myValue, setMyValue] = useState(value ?? '');

  const handleSingleUpload = (file) => {
    const { name } = file;

    const reader = new FileReader();
    reader.onload = (evt) => {
      if (!evt?.target?.result) return;

      const { result } = evt.target;

      const value = result + `,filename:${name}`;

      setMyValue(value);
      onChange(value);
    };
    reader.readAsDataURL(file);
  };

  const handleMultipleUpload = async (files) => {
    const promises = files.map((file) => {
      const reader = new FileReader();
      const { name } = file;

      return new Promise((resolve) => {
        reader.onload = (evt) => {
          if (!evt?.target?.result) return;

          const { result } = evt.target;
          const value = result + `,filename:${name}`;
          resolve(value);
        };
        reader.readAsDataURL(file);
      });
    });

    const results = await Promise.all(promises);
    setMyValue(JSON.stringify(results));
    onChange(JSON.stringify(results));
  };

  const handleFileUpload = async (e) => {
    if (!e.target.files) return;

    const { files } = e.target;
    const numFiles = files.length;

    if (numFiles === 1) {
      handleSingleUpload(files[0]);
    } else if (numFiles > 1) {
      await handleMultipleUpload(Array.from(files));
    }
  };

  // Render file input field and current file name (if applicable)
  return (
    <FormControl sx={{ mt: 1, width: '100%' }} size="small">
      <span
        style={{
          fontStyle: 'italic',
          color: theme.palette.grey['800'],
          marginBottom: '1rem',
        }}
      >
        {myValue ? getFileName(myValue) : 'Choose a file to upload'}
      </span>
      <Button
        disabled={disabled}
        variant="outlined"
        component="label"
        fullWidth
        startIcon={<IconUpload />}
        sx={{ marginRight: '1rem' }}
      >
        {'Upload File'}
        <input type="file" multiple accept={fileType} hidden onChange={handleFileUpload} />
      </Button>
    </FormControl>
  );
};

File.propTypes = {
  value: PropTypes.string,
  fileType: PropTypes.string,
  onChange: PropTypes.func,
  disabled: PropTypes.bool,
};

export default File;

