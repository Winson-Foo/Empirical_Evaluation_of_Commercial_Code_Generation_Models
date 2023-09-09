// ParentComponent.js
import ConfirmDialog from './ConfirmDialog'

const ParentComponent = () => {
  const { confirmState, showConfirmDialog, hideConfirmDialog } = useConfirm()

  const handleDeleteClick = (id) => {
    showConfirmDialog(
      'Delete Confirmation',
      `Are you sure you want to delete item ${id}?`,
      'Cancel',
      'Delete',
      hideConfirmDialog,
      () => {
        // Delete logic goes here
        hideConfirmDialog()
      }
    )
  }

  return (
    <>
      {list.map((item) => (
        <ListItem key={item.id}>
          <ListItemText primary={item.name} />
          <IconButton aria-label='delete' onClick={() => handleDeleteClick(item.id)}>
            <DeleteIcon />
          </IconButton>
        </ListItem>
      ))}
      <ConfirmDialog confirmState={confirmState} hideConfirmDialog={hideConfirmDialog} />
    </>
  )
}

export default ParentComponent