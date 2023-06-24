/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.uiapp;

import android.os.Bundle;
import androidx.annotation.Nullable;

import com.facebook.react.ReactActivity;
import com.facebook.react.ReactActivityDelegate;
import com.facebook.react.defaults.DefaultNewArchitectureEntryPoint;
import com.facebook.react.defaults.DefaultReactActivityDelegate;

public class RNTesterActivity extends ReactActivity {

  private static final String PARAM_ROUTE = "route";

  /**
   * Custom delegate that extends DefaultReactActivityDelegate and sets initial props
   * based on the route passed via intent extras.
   */
  public static class RNTesterActivityDelegate extends DefaultReactActivityDelegate {

    private Bundle mInitialProps = null;
    private final @Nullable ReactActivity mActivity;

    public RNTesterActivityDelegate(ReactActivity activity, String mainComponentName) {
      // Pass required params to parent constructor.
      super(activity, mainComponentName, DefaultNewArchitectureEntryPoint.getFabricEnabled());
      this.mActivity = activity;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
      // Get remote param before calling super which uses it.
      Bundle extras = mActivity.getIntent().getExtras();
      if (extras != null && extras.containsKey(PARAM_ROUTE)) {
        // Build the initial props bundle if route param exists.
        String routeUri = new StringBuilder("rntester://example/")
                .append(extras.getString(PARAM_ROUTE))
                .append("Example")
                .toString();
        mInitialProps = new Bundle();
        mInitialProps.putString("exampleFromAppetizeParams", routeUri);
      }
      super.onCreate(savedInstanceState);
    }

    @Override
    protected Bundle getLaunchOptions() {
      // Return the initial props bundle.
      return mInitialProps;
    }
  }

  @Override
  protected ReactActivityDelegate createReactActivityDelegate() {
    // Return the custom delegate.
    return new RNTesterActivityDelegate(this, getMainComponentName());
  }

  @Override
  protected String getMainComponentName() {
    // Override the main component name defined in the AndroidManifest.xml file.
    return "RNTesterApp";
  }
} 