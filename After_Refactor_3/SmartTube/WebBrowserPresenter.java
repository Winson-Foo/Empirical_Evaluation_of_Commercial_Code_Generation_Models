package com.liskovsoft.smartyoutubetv2.common.app.presenters;

import android.content.Context;
import com.liskovsoft.smartyoutubetv2.common.app.presenters.base.BasePresenter;
import com.liskovsoft.smartyoutubetv2.common.app.views.ViewManager;
import com.liskovsoft.smartyoutubetv2.common.app.views.WebBrowserView;

public class WebBrowserPresenter extends BasePresenter<WebBrowserView> {

    private final ViewManager mViewManager;
    private String mUrl;

    public WebBrowserPresenter(Context context) {
        super(context);
        mViewManager = ViewManager.instance(context);
    }

    public void loadUrl(String url) {
        mUrl = url;
        mViewManager.startView(WebBrowserView.class);
        getView().loadUrl(url);
    }

    public static void unhold() {
        sInstance = null;
    }

} 
