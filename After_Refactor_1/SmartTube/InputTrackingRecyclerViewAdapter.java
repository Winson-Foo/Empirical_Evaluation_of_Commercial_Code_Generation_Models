public abstract class InputTrackingRecyclerViewAdapter<VH extends RecyclerView.ViewHolder> extends RecyclerView.Adapter<VH> {
    private SelectionTracker mSelectionTracker; 

    @Override
    public void onAttachedToRecyclerView(final RecyclerView recyclerView) {
        super.onAttachedToRecyclerView(recyclerView);

        recyclerView.setOnKeyListener(new KeyEventHandler(mSelectionTracker));
    }

    public void setSelectionTracker(SelectionTracker selectionTracker) {
        mSelectionTracker = selectionTracker;
    }
}

public class SelectionTracker {
    private int mSelectedItem = 0;

    public int getSelectedItem() {
        return mSelectedItem;
    }

    public void setSelectedItem(int selectedItem) {
        mSelectedItem = selectedItem;
    }

    public boolean tryMoveSelection(int direction, RecyclerView.Adapter adapter, RecyclerView.LayoutManager layoutManager) {
        int nextSelectItem = mSelectedItem + direction;

        if (nextSelectItem >= 0 && nextSelectItem < adapter.getItemCount()) {
            adapter.notifyItemChanged(mSelectedItem);
            mSelectedItem = nextSelectItem;
            adapter.notifyItemChanged(mSelectedItem);
            layoutManager.smoothScrollToPosition(mSelectedItem);
            return true;
        }
        return false;
    }
}

public class KeyEventHandler implements View.OnKeyListener {
    private final SelectionTracker mSelectionTracker;

    public KeyEventHandler(SelectionTracker selectionTracker) {
        mSelectionTracker = selectionTracker;
    }

    @Override
    public boolean onKey(View v, int keyCode, KeyEvent event) {
        RecyclerView.LayoutManager lm = ((RecyclerView) v).getLayoutManager();

        if (event.getAction() == KeyEvent.ACTION_DOWN) {
            if (InputTrackingRecyclerViewAdapter.isConfirmButton(event)) {
                handleConfirmButton(event);
                return true;
            } else if (keyCode == KeyEvent.KEYCODE_DPAD_DOWN) {
                return mSelectionTracker.tryMoveSelection(1, ((RecyclerView) v).getAdapter(), lm);
            } else if (keyCode == KeyEvent.KEYCODE_DPAD_UP) {
                return mSelectionTracker.tryMoveSelection(-1, ((RecyclerView) v).getAdapter(), lm);
            }
        } else if (event.getAction() == KeyEvent.ACTION_UP && InputTrackingRecyclerViewAdapter.isConfirmButton(event) && ((event.getFlags() & KeyEvent.FLAG_LONG_PRESS) != KeyEvent.FLAG_LONG_PRESS)) {
            ((RecyclerView) v).findViewHolderForAdapterPosition(mSelectionTracker.getSelectedItem()).itemView.performClick();
            return true;
        }
        return false;
    }

    private void handleConfirmButton(KeyEvent event) {
        if ((event.getFlags() & KeyEvent.FLAG_LONG_PRESS) == KeyEvent.FLAG_LONG_PRESS) {
            // handle long press
        } else {
            // handle short press
        }
    }
}