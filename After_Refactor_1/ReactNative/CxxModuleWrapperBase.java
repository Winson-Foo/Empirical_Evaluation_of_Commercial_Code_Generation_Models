package com.facebook.react.bridge;

import com.facebook.jni.HybridData;
import com.facebook.proguard.annotations.DoNotStrip;

/**
 * A Java Object which represents a cross-platform module
 *
 * <p>This module implements the NativeModule interface and provides
 * common methods and properties for all modules.
 */
public abstract class BaseModule implements NativeModule {

  @DoNotStrip
  protected HybridData hybridData;

  @Override
  public abstract String getName();

  @Override
  public void initialize() {
    // do nothing
  }

  @Override
  public boolean canOverrideExistingModule() {
    return false;
  }

  @Override
  public void onCatalystInstanceDestroy() {}

  @Override
  public void invalidate() {
    hybridData.resetNative();
  }
}
```

CxxModuleWrapperBase class:

```
package com.facebook.react.bridge;

import com.facebook.jni.HybridData;
import com.facebook.proguard.annotations.DoNotStrip;

/**
 * A Java Object which represents a cross-platform C++ module
 *
 * <p>This module implements the NativeModule interface but will never be invoked from Java,
 * instead the underlying Cxx module will be extracted by the bridge and called directly.
 */
@DoNotStrip
public class CxxModuleWrapperBase extends BaseModule {

  static {
    ReactBridge.staticInit();
  }

  // For creating a wrapper from C++, or from a derived class.
  protected CxxModuleWrapperBase(HybridData hd) {
    super();
    hybridData = hd;
  }

  // Replace the current native module held by this wrapper by a new instance
  protected void resetModule(HybridData hd) {
    if (hd != hybridData) {
      hybridData.resetNative();
      hybridData = hd;
    }
  }

  @Override
  public String getName() {
    // return the name of the module
    return null;
  }
}