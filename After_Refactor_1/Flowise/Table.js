import PropTypes from 'prop-types'
import { TableContainer, Table, TableHead, TableCell, TableRow, TableBody, Paper } from '@mui/material'

const TableHeader = ({ columns }) => {
    return (
        <TableHead>
            <TableRow>
                {columns.map((col, index) => (
                    <TableCell key={index}>{col.charAt(0).toUpperCase() + col.slice(1)}</TableCell>
                ))}
            </TableRow>
        </TableHead>
    )
}

const TableRow = ({ row }) => {
    return (
        <TableRow sx={{ '&:last-child td, &:last-child th': { border: 0 } }}>
            {Object.keys(row).map((key, index) => (
                <TableCell key={index}>{row[key]}</TableCell>
            ))}
        </TableRow>
    )
}

const TableViewOnly = ({ columns = [], rows = [] }) => {
    return (
        <TableContainer component={Paper}>
            <Table sx={{ minWidth: 650 }} aria-label='simple table'>
                <TableHeader columns={columns} />
                <TableBody>
                    {rows.map((row, index) => (
                        <TableRow key={index} row={row} />
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
    )
}

TableHeader.propTypes = {
    columns: PropTypes.array.isRequired
}

TableRow.propTypes = {
    row: PropTypes.object.isRequired
}

TableViewOnly.propTypes = {
    rows: PropTypes.array,
    columns: PropTypes.array
}

export default TableViewOnly;

