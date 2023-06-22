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
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;

import javax.inject.Provider;

public abstract class TurboReactPackage implements ReactPackage {

    @Override
    public List<NativeModule> createNativeModules(ReactApplicationContext reactContext) {
        throw new UnsupportedOperationException("createNativeModules is not supported.");
    }

    public abstract @Nullable NativeModule getModule(String name, final ReactApplicationContext reactContext);

    public Iterable<ModuleHolder> getModuleHolderIterator(final ReactApplicationContext reactContext) {
        Set<ReactModuleInfo> reactModuleInfos = getReactModuleInfoProvider().getReactModuleInfos().values();
        List<ModuleHolder> moduleHolders = new ArrayList<>();

        for (ReactModuleInfo info : reactModuleInfos) {
            if (ReactFeatureFlags.useTurboModules && info.isTurboModule()) {
                continue;
            }
            ModuleHolder moduleHolder = new ModuleHolder(info, new ModuleHolderProvider(info.name(), reactContext));
            moduleHolders.add(moduleHolder);
        }
        return moduleHolders;
    }

    protected List<ModuleSpec> getViewManagers(ReactApplicationContext reactContext) {
        return Collections.emptyList();
    }

    @Override
    public List<ViewManager> createViewManagers(ReactApplicationContext reactContext) {
        List<ModuleSpec> viewManagerModuleSpecs = getViewManagers(reactContext);
        if (viewManagerModuleSpecs.isEmpty()) {
            return Collections.emptyList();
        }

        List<ViewManager> viewManagers = new ArrayList<>();
        for (ModuleSpec spec : viewManagerModuleSpecs) {
            Provider provider = spec.getProvider();
            viewManagers.add((ViewManager) provider.get());
        }
        return viewManagers;
    }

    public abstract ReactModuleInfoProvider getReactModuleInfoProvider();

    private class ModuleHolderProvider implements Provider<NativeModule> {

        private final String name;
        private final ReactApplicationContext reactContext;

        public ModuleHolderProvider(String name, ReactApplicationContext reactContext) {
            this.name = name;
            this.reactContext = reactContext;
        }

        @Override
        public NativeModule get() {
            return getModule(name, reactContext);
        }
    }
}