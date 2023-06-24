public class AddDevicePresenter extends BasePresenter<AddDeviceView> {
    private static final String TAG = AddDevicePresenter.class.getSimpleName();
    private final MediaService mMediaService;
    private Disposable mDeviceCodeAction;

    public AddDevicePresenter(Context context, MediaService mediaService) {
        super(context);
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

    public void start() {
        RxHelper.disposeActions(mDeviceCodeAction);
        ViewManager.instance(getContext()).startView(AddDeviceView.class);
    }
} 