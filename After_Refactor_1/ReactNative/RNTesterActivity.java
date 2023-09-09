package com.facebook.react.uiapp;

import android.os.Bundle;
import androidx.annotation.Nullable;
import com.facebook.react.ReactActivity;
import com.facebook.react.ReactActivityDelegate;
import com.facebook.react.defaults.DefaultNewArchitectureEntryPoint;
import com.facebook.react.defaults.DefaultReactActivityDelegate;

public class RNTesterActivity extends ReactActivity {
    
    public static class RNTesterActivityDelegate extends DefaultReactActivityDelegate {
        
        private static final String PARAM_ROUTE = "route";
        private Bundle mInitialProps = null;
        private final @Nullable ReactActivity mActivity;
    
        public RNTesterActivityDelegate(ReactActivity activity, String mainComponentName) {
            super(activity, mainComponentName, DefaultNewArchitectureEntryPoint.getFabricEnabled());
            this.mActivity = activity;
        }
        
        @Override
        protected void onCreate(Bundle savedInstanceState) {
            setInitialProps();
            super.onCreate(savedInstanceState);
        }

        private void setInitialProps() {
            Bundle bundle = mActivity.getIntent().getExtras();
            if (bundle != null && bundle.containsKey(PARAM_ROUTE)) {
                String routeUri =
                       createExampleUri(bundle.getString(PARAM_ROUTE));
                mInitialProps = new Bundle();
                mInitialProps.putString("exampleFromAppetizeParams", routeUri);
            }
        }

        private String createExampleUri(String route) {
            return new StringBuilder("rntester://example/")
                    .append(route)
                    .append("Example")
                    .toString();
        }
        
        @Override
        protected Bundle getLaunchOptions() {
            return mInitialProps;
        }
    }

    @Override
    protected ReactActivityDelegate createReactActivityDelegate() {
        return new RNTesterActivityDelegate(this, getMainComponentName());
    }

    @Override
    protected String getMainComponentName() {
        return "RNTesterApp";
    }

}