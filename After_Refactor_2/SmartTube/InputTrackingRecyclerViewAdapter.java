package com.stfalcon.chatkit.commons;

import android.view.KeyEvent;
import android.view.View;
import android.widget.Toast;

import androidx.recyclerview.widget.RecyclerView;

public abstract class InputTrackingRecyclerViewAdapter<VH extends RecyclerView.ViewHolder> extends RecyclerView.Adapter<VH> {

    private int mSelectedItem = 0;
    private RecyclerView mRecyclerView;

    @Override
    public void onAttachedToRecyclerView(final RecyclerView recyclerView) {
        super.onAttachedToRecyclerView(recyclerView);

        mRecyclerView = recyclerView;

        recyclerView.setOnKeyListener(new View.OnKeyListener() {
            @Override
            public boolean onKey(View v, int keyCode, KeyEvent event) {
                if (event.getAction() == KeyEvent.ACTION_DOWN) {
                    if (isConfirmButton(event)) {
                        return confirmButtonClicked();
                    } else {
                        if (keyCode == KeyEvent.KEYCODE_DPAD_DOWN) {
                            return moveSelection(1);
                        } else if (keyCode == KeyEvent.KEYCODE_DPAD_UP) {
                            return moveSelection(-1);
                        }
                    }
                } else if (event.getAction() == KeyEvent.ACTION_UP && isConfirmButton(event) && ((event.getFlags() & KeyEvent.FLAG_LONG_PRESS) != KeyEvent.FLAG_LONG_PRESS)) {
                    return itemClicked();
                }
                return false;
            }
        });

    }

    private boolean confirmButtonClicked(){
        if ((event.getFlags() & KeyEvent.FLAG_LONG_PRESS) == KeyEvent.FLAG_LONG_PRESS) {
            mRecyclerView.findViewHolderForAdapterPosition(mSelectedItem).itemView.performLongClick();
        } else {
            event.startTracking();
        }
        return true;
    }

    private boolean itemClicked(){
        mRecyclerView.findViewHolderForAdapterPosition(mSelectedItem).itemView.performClick();
        return true;
    }

    private boolean moveSelection(int direction) {
        int nextSelectItem = mSelectedItem + direction;

        if (nextSelectItem >= 0 && nextSelectItem < getItemCount()) {
            notifyItemChanged(mSelectedItem);
            mSelectedItem = nextSelectItem;
            notifyItemChanged(mSelectedItem);
            mRecyclerView.smoothScrollToPosition(mSelectedItem);
            return true;
        }

        return false;
    }

    public int getSelectedItem() {
        return mSelectedItem;
    }

    public void setSelectedItem(int selectedItem) {
        mSelectedItem = selectedItem;
    }

    public RecyclerView getRecyclerView() {
        return mRecyclerView;
    }

    public static boolean isConfirmButton(KeyEvent event) {
        return event.getKeyCode() == KeyEvent.KEYCODE_ENTER || event.getKeyCode() == KeyEvent.KEYCODE_DPAD_CENTER || event.getKeyCode() == KeyEvent.KEYCODE_BUTTON_A;
    }

} 