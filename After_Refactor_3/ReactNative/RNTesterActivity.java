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

  public static class RNTesterActivityDelegate extends DefaultReactActivityDelegate {

    private static final String PARAM_ROUTE = "route";
    private static final String EXAMPLE_PREFIX = "rntester://example/";
    private static final String EXAMPLE_SUFFIX = "Example";
    private static final String EXAMPLE_PARAM_KEY = "exampleFromAppetizeParams";

    private final ReactActivity mActivity;
    private Bundle mInitialProps = null;

    public RNTesterActivityDelegate(ReactActivity activity, String mainComponentName) {
      super(activity, mainComponentName, DefaultNewArchitectureEntryPoint.getFabricEnabled());
      this.mActivity = activity;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
      super.onCreate(savedInstanceState);
      mInitialProps = getExampleParamsFromIntent(mActivity.getIntent().getExtras());
    }

    @Override
    protected Bundle getLaunchOptions() {
      return mInitialProps;
    }

    private Bundle getExampleParamsFromIntent(@Nullable Bundle extras) {
      if (extras == null || !extras.containsKey(PARAM_ROUTE)) {
        return null;
      }
      String route = extras.getString(PARAM_ROUTE);
      if (route == null || route.isEmpty()) {
        return null;
      }
      String example = EXAMPLE_PREFIX + route + EXAMPLE_SUFFIX;
      Bundle exampleParams = new Bundle();
      exampleParams.putString(EXAMPLE_PARAM_KEY, example);
      return exampleParams;
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