package com.liskovsoft.smartyoutubetv2.common.app.presenters;

import android.content.Context;
import com.liskovsoft.mediaserviceinterfaces.MediaService;
import com.liskovsoft.sharedutils.mylogger.Log;
import com.liskovsoft.smartyoutubetv2.common.app.presenters.base.BasePresenter;
import com.liskovsoft.smartyoutubetv2.common.app.views.AddDeviceView;
import com.liskovsoft.smartyoutubetv2.common.app.views.ViewManager;
import com.liskovsoft.sharedutils.rx.RxHelper;
import com.liskovsoft.youtubeapi.service.YouTubeMediaService;
import io.reactivex.disposables.CompositeDisposable;

public class AddDevicePresenter extends BasePresenter<AddDeviceView> {
    private static final String TAG = AddDevicePresenter.class.getSimpleName();
    private static final String PAIRING_PROCESS_TAG = "AddDevicePresenter::PairingProcess";
    private final MediaService mMediaService;
    private CompositeDisposable mCompositeDisposable;

    private AddDevicePresenter(MediaService mediaService, Context context) {
        super(context);
        mMediaService = mediaService;
        mCompositeDisposable = new CompositeDisposable();
    }

    public static AddDevicePresenter instance(Context context) {
        if (sInstance == null) {
            sInstance = new AddDevicePresenter(YouTubeMediaService.instance(), context);
        }

        sInstance.setContext(context);

        return sInstance;
    }

    @Override
    public void onViewDestroyed() {
        mCompositeDisposable.dispose();
        super.onViewDestroyed();
    }

    @Override
    public void onViewInitialized() {
        startPairingProcess();
    }

    public void onActionClicked() {
        getView().close();
    }

    private void startPairingProcess() {
        mCompositeDisposable.add(mMediaService.getRemoteControlService().getPairingCodeObserve()
                .subscribe(
                        deviceCode -> getView().showCode(deviceCode),
                        error -> Log.e(PAIRING_PROCESS_TAG, "Get pairing code error: %s", error.getMessage())
                ));
    }

    public void start() {
        mCompositeDisposable.dispose();
        ViewManager.instance(getContext()).startView(AddDeviceView.class);
    }
}