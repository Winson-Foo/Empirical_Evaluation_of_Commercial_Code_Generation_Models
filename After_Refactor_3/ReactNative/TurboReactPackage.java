package com.facebook.react;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import com.facebook.react.bridge.ModuleHolder;
import com.facebook.react.bridge.ModuleSpec;
import com.facebook.react.bridge.NativeModule;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.config.ReactFeatureFlags;
import com.facebook.react.module.model.ReactModuleInfo;
import com.facebook.react.module.model.ReactModuleInfoProvider;
import com.facebook.react.uimanager.ViewManager;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;
import javax.inject.Provider;

public abstract class TurboReactPackage implements ReactPackage {

  public List<NativeModule> createNativeModules(ReactApplicationContext reactContext) {
    throw new UnsupportedOperationException("Incompatible with TurboModules. Please use getModuleList() or getModule() instead.");
  }

  public abstract @Nullable NativeModule getModule(String name, final ReactApplicationContext reactContext);

  public Iterable<ModuleHolder> getNativeModuleIterator(final ReactApplicationContext reactContext) {
    final Set<Map.Entry<String, ReactModuleInfo>> entrySet = getReactModuleInfoProvider().getReactModuleInfos().entrySet();
    final Iterator<Map.Entry<String, ReactModuleInfo>> entrySetIterator = entrySet.iterator();

    return new Iterable<ModuleHolder>() {
      @NonNull
      @Override
      public Iterator<ModuleHolder> iterator() {
        return new Iterator<ModuleHolder>() {
          Map.Entry<String, ReactModuleInfo> nextEntry = null;

          private void findNext() {
            while (entrySetIterator.hasNext()) {
              Map.Entry<String, ReactModuleInfo> entry = entrySetIterator.next();
              ReactModuleInfo reactModuleInfo = entry.getValue();

              boolean skipIteration = ReactFeatureFlags.useTurboModules && reactModuleInfo.isTurboModule(); // skip iteration if it's a TurboModule and TurboModules are enabled

              if (!skipIteration) {
                nextEntry = entry;
                break;
              }
            }

            if (nextEntry == null) {
              throw new NoSuchElementException("ModuleHolder not found");
            }
          }

          @Override
          public boolean hasNext() {
            if (nextEntry == null) {
              findNext();
            }
            return nextEntry != null;
          }

          @Override
          public ModuleHolder next() {
            if (nextEntry == null) {
              findNext();
            }

            Map.Entry<String, ReactModuleInfo> entry = nextEntry;
            String name = entry.getKey();
            ReactModuleInfo reactModuleInfo = entry.getValue();

            // Advance iterator
            nextEntry = null;

            return new ModuleHolder(reactModuleInfo, new ModuleHolderProvider(name, reactContext));
          }

          @Override
          public void remove() {
            throw new UnsupportedOperationException("Cannot remove native modules from the list");
          }
        };
      }
    };
  }

  protected List<ModuleSpec> getViewManagers(ReactApplicationContext reactContext) {
    return Collections.emptyList();
  }

  @Override
  public List<ViewManager> createViewManagers(ReactApplicationContext reactContext) {
    List<ViewManager> viewManagers = new ArrayList<>();
    List<ModuleSpec> moduleSpecs = getViewManagers(reactContext);

    for (ModuleSpec spec : moduleSpecs) {
      viewManagers.add((ViewManager) spec.getProvider().get());
    }

    return viewManagers;
  }

  public abstract ReactModuleInfoProvider getReactModuleInfoProvider();

  private class ModuleHolderProvider implements Provider<NativeModule> {

    private final String mName;
    private final ReactApplicationContext mReactContext;

    public ModuleHolderProvider(String name, ReactApplicationContext reactContext) {
      mName = name;
      mReactContext = reactContext;
    }

    @Override
    public NativeModule get() {
      return getModule(mName, mReactContext);
    }
  }
} 