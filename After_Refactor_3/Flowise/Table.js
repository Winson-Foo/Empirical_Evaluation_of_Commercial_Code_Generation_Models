import PropTypes from 'prop-types'
import { TableContainer, Table, TableHead, TableCell, TableRow, TableBody, Paper } from '@mui/material'

const capitalize = (str) => {
  // Capitalize the first letter of a string
  return str.charAt(0).toUpperCase() + str.slice(1)
}

const renderTableHeader = (columns) => {
  // Render the table header based on the columns prop
  return (
    <TableHead>
      <TableRow>
        {columns.map((col, index) => (
          <TableCell key={index}>{capitalize(col)}</TableCell>
        ))}
      </TableRow>
    </TableHead>
  )
}

const renderTableRow = (row, index) => {
  // Render a single table row based on the row data
  const cells = Object.values(row).map((value, index) => (
    <TableCell key={index}>{value}</TableCell>
  ))
  return (
    <TableRow key={index} sx={{ '&:last-child td, &:last-child th': { border: 0 } }}>
      {cells}
    </TableRow>
  )
}

export const TableViewOnly = ({ columns, rows }) => {
  // Render the table using the provided columns and rows props
  return (
    <TableContainer component={Paper}>
      <Table sx={{ minWidth: 650 }} aria-label='simple table'>
        {renderTableHeader(columns)}
        <TableBody>
          {rows.map((row, index) => renderTableRow(row, index))}
        </TableBody>
      </Table>
    </TableContainer>
  )
}

TableViewOnly.propTypes = {
  rows: PropTypes.array,
  columns: PropTypes.array
}

