import PropTypes from 'prop-types'
import { TableContainer, Table, Paper } from '@mui/material'
import { TableHead } from './TableHead'
import { TableRow } from './TableRow'

export const TableViewOnly = ({ columns = [], rows = [] }) => {
  return (
    <TableContainer component={Paper}>
      <Table sx={{ minWidth: 650 }} aria-label='simple table'>
        <TableHead columns={columns} />
        <tbody>
          {rows.map((row) => (
            <TableRow key={row.id} row={row} />
          ))}
        </tbody>
      </Table>
    </TableContainer>
  )
}

TableViewOnly.propTypes = {
  rows: PropTypes.arrayOf(PropTypes.object),
  columns: PropTypes.arrayOf(PropTypes.string),
}

// Example usage:
// <TableViewOnly
//   columns={['id', 'name', 'age']}
//   rows={[
//     { id: 1, name: 'Alice', age: 30 },
//     { id: 2, name: 'Bob', age: 25 },
//     { id: 3, name: 'Charlie', age: 40 },
//   ]}
// />

// TableHead.js
import PropTypes from 'prop-types'
import { TableHead as MUITableHead, TableRow, TableCell } from '@mui/material'

export const TableHead = ({ columns }) => {
  const capitalize = (str) => str.charAt(0).toUpperCase() + str.slice(1)

  return (
    <MUITableHead>
      <TableRow>
        {columns.map((column) => (
          <TableCell key={column}>{capitalize(column)}</TableCell>
        ))}
      </TableRow>
    </MUITableHead>
  )
}

TableHead.propTypes = {
  columns: PropTypes.arrayOf(PropTypes.string),
}

// TableRow.js
import PropTypes from 'prop-types'
import { TableRow as MUITableRow, TableCell } from '@mui/material'

export const TableRow = ({ row }) => {
  return (
    <MUITableRow>
      {Object.values(row).map((value, index) => (
        <TableCell key={index}>{value}</TableCell>
      ))}
    </MUITableRow>
  )
}

TableRow.propTypes = {
  row: PropTypes.object,
}

