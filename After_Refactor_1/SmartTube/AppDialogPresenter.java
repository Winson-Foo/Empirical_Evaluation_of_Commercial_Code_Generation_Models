package com.liskovsoft.smartyoutubetv2.common.app.presenters;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import androidx.annotation.NonNull;
import com.liskovsoft.smartyoutubetv2.common.app.models.playback.ui.OptionItem;
import com.liskovsoft.smartyoutubetv2.common.app.presenters.base.BasePresenter;
import com.liskovsoft.smartyoutubetv2.common.app.views.AppDialogView;
import com.liskovsoft.smartyoutubetv2.common.app.views.ViewManager;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class AppDialogPresenter extends BasePresenter<AppDialogView> {
    private static AppDialogPresenter sInstance;
    private final Runnable mCloseDialog = this::closeDialog;
    private final List<Runnable> mOnFinishCallbacks = new ArrayList<>();
    private String mDialogTitle;
    private long mCloseTimeoutMs;
    private boolean mIsTransparent;
    private List<AppDialogCategory> mCategories = new ArrayList<>();
    private boolean mIsExpandable = true;
    private final Handler mHandler;

    private AppDialogPresenter(@NonNull Context context) {
        super(context);
        mHandler = new Handler(Looper.getMainLooper());
    }

    private void clear() {
        mCloseTimeoutMs = 0;
        mHandler.removeCallbacks(mCloseDialog);
        mCategories = new ArrayList<>();
        mIsExpandable = true;
        mIsTransparent = false;
        mDialogTitle = null;
    }

    private void setupCloseTimeout() {
        mHandler.removeCallbacks(mCloseDialog);
        if (mCloseTimeoutMs > 0) {
            mHandler.postDelayed(mCloseDialog, mCloseTimeoutMs);
        }
    }

    @Override
    public void onFinish() {
        super.onFinish();
        clear();
        for (Runnable callback : mOnFinishCallbacks) {
            if (callback != null) {
                callback.run();
            }
        }
        mOnFinishCallbacks.clear();
    }

    @Override
    public void onViewInitialized() {
        getView().show(mCategories, mDialogTitle, mIsExpandable, mIsTransparent);
        mCategories = new ArrayList<>();
        mIsExpandable = true;
        mIsTransparent = false;
    }

    @Override
    public void onViewDestroyed() {
        super.onViewDestroyed();
        clear();
    }

    public void showDialog(String dialogTitle, List<AppDialogCategory> categories, boolean isExpandable, boolean isTransparent, Runnable onClose, long closeTimeoutMs) {
        mDialogTitle = dialogTitle;
        mCategories = categories;
        mIsExpandable = isExpandable;
        mIsTransparent = isTransparent;
        mCloseTimeoutMs = closeTimeoutMs;
        if (getView() != null) {
            onViewInitialized();
        }
        ViewManager.instance(getContext()).startView(AppDialogView.class, true);
        setupCloseTimeout();
        mOnFinishCallbacks.add(onClose);
    }

    public void closeDialog() {
        if (getView() != null) {
            getView().finish();
        }
    }

    public boolean isDialogShown() {
        return ViewManager.isVisible(getView()) || ViewManager.instance(getContext()).isViewPending(AppDialogView.class);
    }

    public void appendCategory(AppDialogCategory category) {
        mCategories.add(category);
    }

    // other functions removed for brevity
}