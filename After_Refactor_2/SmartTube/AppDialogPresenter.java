package com.liskovsoft.smartyoutubetv2.common.app.models.playback.ui;

import java.util.Collections;
import java.util.List;

public class OptionCategory {
    public static OptionCategory radioList(String title, List<OptionItem> items) {
        return new OptionCategory(title, items, TYPE_RADIO_LIST);
    }

    public static OptionCategory checkedList(String title, List<OptionItem> items) {
        return new OptionCategory(title, items, TYPE_CHECKBOX_LIST);
    }

    public static OptionCategory stringList(String title, List<OptionItem> items) {
        return new OptionCategory(title, items, TYPE_STRING_LIST);
    }

    public static OptionCategory longText(String title, OptionItem item) {
        return new OptionCategory(title, Collections.singletonList(item), TYPE_LONG_TEXT);
    }

    public static OptionCategory chat(String title, OptionItem item) {
        return new OptionCategory(title, Collections.singletonList(item), TYPE_CHAT);
    }

    public static OptionCategory comments(String title, OptionItem item) {
        return new OptionCategory(title, Collections.singletonList(item), TYPE_COMMENTS);
    }

    public static OptionCategory singleSwitch(OptionItem item) {
        return new OptionCategory(null, Collections.singletonList(item), TYPE_SINGLE_SWITCH);
    }

    public static OptionCategory singleButton(OptionItem item) {
        return new OptionCategory(null, Collections.singletonList(item), TYPE_SINGLE_BUTTON);
    }

    private OptionCategory(String title, List<OptionItem> items, int type) {
        this.type = type;
        this.title = title;
        this.items = items;
    }

    public static final int TYPE_RADIO_LIST = 0;
    public static final int TYPE_CHECKBOX_LIST = 1;
    public static final int TYPE_SINGLE_SWITCH = 2;
    public static final int TYPE_SINGLE_BUTTON = 3;
    public static final int TYPE_STRING_LIST = 4;
    public static final int TYPE_LONG_TEXT = 5;
    public static final int TYPE_CHAT = 6;
    public static final int TYPE_COMMENTS = 7;
    public int type;
    public String title;
    public List<OptionItem> items;
}
```

AppDialogPresenter.java

```
package com.liskovsoft.smartyoutubetv2.common.app.presenters;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;

import com.liskovsoft.smartyoutubetv2.common.app.models.playback.ui.OptionCategory;
import com.liskovsoft.smartyoutubetv2.common.app.models.playback.ui.OptionItem;
import com.liskovsoft.smartyoutubetv2.common.app.presenters.base.BasePresenter;
import com.liskovsoft.smartyoutubetv2.common.app.views.AppDialogView;
import com.liskovsoft.smartyoutubetv2.common.app.views.ViewManager;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class AppDialogPresenter extends BasePresenter<AppDialogView> {
    private final Handler mHandler;
    private final Runnable mCloseDialog = this::closeDialog;
    private final List<Runnable> mOnFinishListeners = new ArrayList<>();
    private String mTitle;
    private long mTimeoutMs;
    private boolean mIsTransparent;
    private List<OptionCategory> mOptionCategories;
    private boolean mIsExpandable = true;

    public AppDialogPresenter(Context context, Handler handler) {
        super(context);
        mOptionCategories = new ArrayList<>();
        mHandler = handler;
    }

    public static AppDialogPresenter getInstance(Context context, Handler handler) {
        return new AppDialogPresenter(context, handler);
    }

    @Override
    public void onFinish() {
        super.onFinish();
        clear();

        for (Runnable onFinishListener : mOnFinishListeners) {
            if (onFinishListener != null) {
                onFinishListener.run();
            }
        }

        mOnFinishListeners.clear();
    }

    private void clear() {
        mTimeoutMs = 0;
        mHandler.removeCallbacks(mCloseDialog);
        mOptionCategories = new ArrayList<>();
        mIsExpandable = true;
        mIsTransparent = false;
    }

    @Override
    public void onViewInitialized() {
        getView().show(mOptionCategories, mTitle, mIsExpandable, mIsTransparent);
        mOptionCategories = new ArrayList<>();
        mIsExpandable = true;
        mIsTransparent = false;
    }

    @Override
    public void onViewDestroyed() {
        super.onViewDestroyed();
        clear();
    }

    public void showDialog() {
        showDialog(null, null);
    }

    public void showDialog(String dialogTitle) {
        showDialog(dialogTitle, null);
    }

    public void showDialog(Runnable onFinishListener) {
        showDialog(null, onFinishListener);
    }

    public void showDialog(String dialogTitle, Runnable onFinishListener) {
        mTitle = dialogTitle;
        mOnFinishListeners.add(onFinishListener);

        if (getView() != null) {
            onViewInitialized();
        }

        ViewManager.instance(getContext()).startView(AppDialogView.class, true);

        setupTimeout();
    }

    public void closeDialog() {
        if (getView() != null) {
            getView().finish();
        }
    }

    public void goBack() {
        if (getView() != null) {
            getView().goBack();
        }
    }

    public boolean isDialogShown() {
        return ViewManager.isVisible(getView()) || ViewManager.instance(getContext()).isViewPending(AppDialogView.class);
    }

    public void appendRadioCategory(String categoryTitle, List<OptionItem> items) {
        mOptionCategories.add(OptionCategory.radioList(categoryTitle, items));
    }

    public void appendCheckedCategory(String categoryTitle, List<OptionItem> items) {
        mOptionCategories.add(OptionCategory.checkedList(categoryTitle, items));
    }

    public void appendStringsCategory(String categoryTitle, List<OptionItem> items) {
        mOptionCategories.add(OptionCategory.stringList(categoryTitle, items));
    }

    public void appendLongTextCategory(String categoryTitle, OptionItem item) {
        mOptionCategories.add(OptionCategory.longText(categoryTitle, item));
    }

    public void appendChatCategory(String categoryTitle, OptionItem item) {
        mOptionCategories.add(OptionCategory.chat(categoryTitle, item));
    }

    public void appendCommentsCategory(String categoryTitle, OptionItem item) {
        mOptionCategories.add(OptionCategory.comments(categoryTitle, item));
    }

    public void appendSingleSwitch(OptionItem optionItem) {
        mOptionCategories.add(OptionCategory.singleSwitch(optionItem));
    }

    public void appendSingleButton(OptionItem optionItem) {
        mOptionCategories.add(OptionCategory.singleButton(optionItem));
    }

    public void showDialogMessage(String dialogTitle, Runnable onFinishListener, int timeoutMs) {
        showDialog(dialogTitle, onFinishListener);

        new Handler(Looper.getMainLooper()).postDelayed(() -> {
            if (getView() != null) {
                getView().finish();
            }
        }, timeoutMs);
    }

    public void setCloseTimeoutMs(long timeoutMs) {
        mTimeoutMs = timeoutMs;
    }

    public void enableTransparent(boolean isTransparent) {
        mIsTransparent = isTransparent;
    }

    public boolean isTransparent() {
        return getView() != null && getView().isTransparent();
    }

    public void enableExpandable(boolean isExpandable) {
        mIsExpandable = isExpandable;
    }

    public boolean isEmpty() {
        return mOptionCategories == null || mOptionCategories.isEmpty();
    }

    private void setupTimeout() {
        mHandler.removeCallbacks(mCloseDialog);

        if (mTimeoutMs > 0) {
            mHandler.postDelayed(mCloseDialog, mTimeoutMs);
        }
    }
} 