package com.liskovsoft.smartyoutubetv2.common.app.presenters;

import android.content.Context;
import com.liskovsoft.smartyoutubetv2.common.app.presenters.base.BasePresenter;
import com.liskovsoft.smartyoutubetv2.common.app.views.ViewManager;
import com.liskovsoft.smartyoutubetv2.common.app.views.WebBrowserView;

public class WebBrowserPresenter extends BasePresenter<WebBrowserView> {

    private final ViewManager viewManager;
    private String url;

    public WebBrowserPresenter(Context context, ViewManager viewManager) {
        super(context);
        this.viewManager = viewManager;
    }

    @Override
    public void onViewInitialized() {
        if (url != null && getView() != null) {
            getView().loadUrl(url);
        }
    }

    public void loadUrl(String url) {
        this.url = url;

        if (viewManager != null) {
            viewManager.startView(WebBrowserView.class);
        }

        if (getView() != null) {
            getView().loadUrl(url);
        }
    }
}

