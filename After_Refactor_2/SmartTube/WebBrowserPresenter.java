package com.liskovsoft.smartyoutubetv2.common.app.presenters;

import android.content.Context;
import com.liskovsoft.smartyoutubetv2.common.app.presenters.base.BasePresenter;
import com.liskovsoft.smartyoutubetv2.common.app.views.ViewManager;
import com.liskovsoft.smartyoutubetv2.common.app.views.WebBrowserView;

public class WebBrowserPresenter extends BasePresenter<WebBrowserView> {
    private final ViewManager m viewManager;
    private final String m Url;

    public WebBrowserPresenter(Context context, ViewManager viewManager) {
        super(context);
        m viewManager = viewManager;
    }

    public void initView() {
        if (mUrl != null && getView() != null) {
            getView().loadUrl(mUrl);
        }
    }

    public void loadUrl(String url) {
        mUrl = url;
        m ViewManager.startView(WebBrowserView.class);
        initView();
    }
}

