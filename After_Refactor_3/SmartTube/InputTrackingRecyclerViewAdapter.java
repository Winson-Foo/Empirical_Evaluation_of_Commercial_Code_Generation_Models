package com.stfalcon.chatkit.commons;

import android.view.KeyEvent;
import android.view.View;
import androidx.recyclerview.widget.RecyclerView;

/**
 * Adapter that handles keyboard navigation events for a RecyclerView.
 */
public abstract class KeyboardNavigationAdapter<VH extends RecyclerView.ViewHolder> extends RecyclerView.Adapter<VH> {

    private int selectedItem = 0;
    private RecyclerView recyclerView;

    /**
     * Sets up the adapter to handle key events on the RecyclerView.
     */
    @Override
    public void onAttachedToRecyclerView(final RecyclerView recyclerView) {
        super.onAttachedToRecyclerView(recyclerView);

        this.recyclerView = recyclerView;
        // Handle key up and key down and attempt to move selection
        recyclerView.setOnKeyListener(new View.OnKeyListener() {
            @Override
            public boolean onKey(View view, int keyCode, KeyEvent event) {
                return handleKeyEvent(event, recyclerView.getLayoutManager(), keyCode);
            }
        });
    }

    /**
     * Modifies the selection based on the direction provided, if possible.
     * Notifies the adapter to redraw and scrolls to the new item.
     *
     * @param layoutManager the RecyclerView's layout manager
     * @param direction      the direction to move the selection (-1 for up, 1 for down)
     * @return true if the selection was successfully moved; false otherwise
     */
    private boolean handleSelectionMove(RecyclerView.LayoutManager layoutManager, int direction) {
        int nextSelectedItem = selectedItem + direction;

        // If still within valid bounds, move the selection, notify to redraw, and scroll
        if (nextSelectedItem >= 0 && nextSelectedItem < getItemCount()) {
            notifyItemChanged(selectedItem);
            selectedItem = nextSelectedItem;
            notifyItemChanged(selectedItem);
            recyclerView.smoothScrollToPosition(selectedItem);
            return true;
        }

        return false;
    }

    /**
     * Handles key events for the RecyclerView by attempting to move the selection
     * up/down and handling confirm button presses (enter, dpad center, or button A).
     *
     * @param event        the key event that occurred
     * @param layoutManager the RecyclerView's layout manager
     * @param keyCode      the key code of the key event
     * @return true if the key event was handled; false otherwise
     */
    private boolean handleKeyEvent(KeyEvent event, RecyclerView.LayoutManager layoutManager, int keyCode) {
        if (event.getAction() == KeyEvent.ACTION_DOWN) {
            if (isConfirmButtonPressed(event)) {
                if ((event.getFlags() & KeyEvent.FLAG_LONG_PRESS) == KeyEvent.FLAG_LONG_PRESS) {
                    recyclerView.findViewHolderForAdapterPosition(selectedItem).itemView.performLongClick();
                } else {
                    event.startTracking();
                }
                return true;
            } else {
                if (keyCode == KeyEvent.KEYCODE_DPAD_DOWN) {
                    return handleSelectionMove(layoutManager, 1);
                } else if (keyCode == KeyEvent.KEYCODE_DPAD_UP) {
                    return handleSelectionMove(layoutManager, -1);
                }
            }
        } else if (event.getAction() == KeyEvent.ACTION_UP && isConfirmButtonPressed(event) && ((event.getFlags() & KeyEvent.FLAG_LONG_PRESS) != KeyEvent.FLAG_LONG_PRESS)) {
            recyclerView.findViewHolderForAdapterPosition(selectedItem).itemView.performClick();
            return true;
        }
        return false;
    }

    /**
     * Determines whether the provided key event represents a confirm button press
     * (enter, dpad center, or button A).
     *
     * @param event the key event to check
     * @return true if the event is a confirm button press; false otherwise
     */
    public static boolean isConfirmButtonPressed(KeyEvent event) {
        switch (event.getKeyCode()) {
            case KeyEvent.KEYCODE_ENTER:
            case KeyEvent.KEYCODE_DPAD_CENTER:
            case KeyEvent.KEYCODE_BUTTON_A:
                return true;
            default:
                return false;
        }
    }

    /**
     * Gets the position of the currently selected item.
     *
     * @return the position of the selected item
     */
    public int getSelectedItemPosition() {
        return selectedItem;
    }

    /**
     * Sets the position of the currently selected item.
     *
     * @param selectedItem the position of the item to select
     */
    public void setSelectedItemPosition(int selectedItem) {
        this.selectedItem = selectedItem;
    }

    /**
     * Binds the view holder to the item at the specified position.
     * This method just calls the abstract onBindViewHolder method,
     * which must be implemented in a subclass.
     *
     * @param holder   the view holder to bind
     * @param position the position of the item to bind to
     */
    @Override
    public void onBindViewHolder(VH holder, int position) {
        onBindViewHolder(holder, position);
    }

}

