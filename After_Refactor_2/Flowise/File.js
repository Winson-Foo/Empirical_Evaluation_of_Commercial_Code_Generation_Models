import { useState } from 'react'
import PropTypes from 'prop-types'
import { useTheme } from '@mui/material/styles'
import { FormControl, Button } from '@mui/material'
import { IconUpload } from '@tabler/icons'
import { getFileName } from 'utils/genericHelper'

/**
 * A component that allows the user to upload one or more files.
 */
export const FileUpload = ({ value, fileType, onChange, disabled = false }) => {
  const theme = useTheme()

  // Use a more descriptive variable name
  const [uploadedFiles, setUploadedFiles] = useState(value ?? '')

  /**
   * Handles the file upload event.
   * @param {*} e The file upload event.
   */
  const handleFileUpload = async (e) => {
    if (!e.target.files) return

    const files = e.target.files

    if (files.length === 1) {
      // Upload a single file
      const file = files[0]
      const { name } = file

      const value = await getBase64(file) + `,filename:${name}`

      setUploadedFiles(value)
      onChange(value)
    } else if (files.length > 1) {
      // Upload multiple files
      const fileValues = await Promise.all(
        Array.from(files).map(async (file) => {
          const { name } = file
          const value = await getBase64(file) + `,filename:${name}`
          return value
        })
      )

      setUploadedFiles(JSON.stringify(fileValues))
      onChange(JSON.stringify(fileValues))
    }
  }

  /**
   * Converts a file to a base64 string.
   * @param {*} file The file to convert.
   * @returns The base64 string representation of the file.
   */
  const getBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.readAsDataURL(file)
      reader.onload = () => resolve(reader.result)
      reader.onerror = (error) => reject(error)
    })
  }

  return (
    <FormControl sx={{ mt: 1, width: '100%' }} size='small'>
      <span
        style={{
          fontStyle: 'italic',
          color: theme.palette.grey['800'],
          marginBottom: '1rem'
        }}
      >
        {uploadedFiles ? getFileName(uploadedFiles) : 'Choose a file to upload'}
      </span>
      <Button
        disabled={disabled}
        variant='outlined'
        component='label'
        fullWidth
        startIcon={<IconUpload />}
        sx={{ marginRight: '1rem' }}
      >
        {'Upload File'}
        <input type='file' multiple accept={fileType} hidden onChange={(e) => handleFileUpload(e)} />
      </Button>
    </FormControl>
  )
}

FileUpload.propTypes = {
  value: PropTypes.string,
  fileType: PropTypes.string,
  onChange: PropTypes.func,
  disabled: PropTypes.bool
}