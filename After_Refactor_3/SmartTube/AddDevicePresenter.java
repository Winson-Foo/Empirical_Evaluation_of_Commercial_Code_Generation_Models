package com.liskovsoft.smartyoutubetv2.common.app.presenters;

import android.content.Context;

import com.liskovsoft.mediaserviceinterfaces.MediaService;
import com.liskovsoft.sharedutils.mylogger.Log;
import com.liskovsoft.smartyoutubetv2.common.app.presenters.base.BasePresenter;
import com.liskovsoft.smartyoutubetv2.common.app.views.AddDeviceView;
import com.liskovsoft.smartyoutubetv2.common.app.views.ViewManager;
import com.liskovsoft.sharedutils.rx.RxHelper;
import com.liskovsoft.youtubeapi.service.YouTubeMediaService;

import io.reactivex.disposables.Disposable;

public class AddDevicePresenter extends BasePresenter<AddDeviceView> {
    private static final String TAG = AddDevicePresenter.class.getSimpleName();
    private final MediaService mMediaService;
    private Disposable mDeviceCodeAction;

    public AddDevicePresenter(MediaService mediaService, AddDeviceView view) {
        super(view.getContext());
        mMediaService = mediaService;
    }

    public void unhold() {
        RxHelper.disposeActions(mDeviceCodeAction);
    }

    @Override
    public void onViewDestroyed() {
        super.onViewDestroyed();
        unhold();
    }

    @Override
    public void onViewInitialized() {
        RxHelper.disposeActions(mDeviceCodeAction);
        updateDeviceCode();
    }

    public void onActionClicked() {
        getView().close();
    }

    private void updateDeviceCode() {
        mDeviceCodeAction = mMediaService.getRemoteControlService().getPairingCodeObserve()
                .subscribe(
                        deviceCode -> getView().showCode(deviceCode),
                        error -> Log.e(TAG, "Get pairing code error: %s", error.getMessage())
                );
    }

    public void start(AddDeviceView view) {
        RxHelper.disposeActions(mDeviceCodeAction);
        ViewManager.instance(view.getContext()).startView(AddDeviceView.class, view);
    }
} 