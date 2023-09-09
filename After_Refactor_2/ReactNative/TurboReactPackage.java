package com.example.react;

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

/**
 * A base class for React Native packages that use TurboModules.
 */
public abstract class TurboModulesPackage implements ReactPackage {

  // Create Native Modules is not supported in TurboModules.
  @Override
  public List<NativeModule> createNativeModules(ReactApplicationContext reactContext) {
    throw new UnsupportedOperationException(
        "In case of TurboModules, createNativeModules is not supported. NativeModuleRegistry should instead use getModuleList or getModule method");
  }

  /**
   * Returns a provider for the given module name, or null if the module is not found.
   * This method is used by the NativeModuleRegistry to create the module list.
   */
  public abstract @Nullable Provider<NativeModule> getModuleProvider(
      String name, final ReactApplicationContext reactContext);

  /**
   * Returns an iterable over all the module specs in this package.
   */
  public Iterable<ModuleHolder> getModuleHolders(
      final ReactApplicationContext reactContext,
      final ReactModuleInfoProvider moduleInfoProvider) {
    return new ModuleHolderIterable(reactContext, moduleInfoProvider, this::getModuleProvider);
  }

  /**
   * Returns a list of module specs for the View Managers provided by this package.
   */
  protected abstract List<ModuleSpec> getViewManagerSpecs(ReactApplicationContext reactContext);

  /**
   * Returns a list of View Managers created by this package.
   */
  @Override
  public final List<ViewManager> createViewManagers(ReactApplicationContext reactContext) {
    List<ModuleSpec> specs = getViewManagerSpecs(reactContext);
    if (specs == null || specs.isEmpty()) {
      return Collections.emptyList();
    }

    List<ViewManager> viewManagers = new ArrayList<>();
    for (ModuleSpec spec : specs) {
      viewManagers.add((ViewManager) spec.getProvider().get());
    }
    return viewManagers;
  }

  /**
   * An implementation of Iterable that provides ModuleHolder objects for all the modules in the package.
   */
  private static class ModuleHolderIterable implements Iterable<ModuleHolder> {

    private final ReactApplicationContext reactContext;
    private final ReactModuleInfoProvider moduleInfoProvider;
    private final ModuleProvider moduleProvider;

    private ModuleHolderIterable(
        ReactApplicationContext reactContext,
        ReactModuleInfoProvider moduleInfoProvider,
        ModuleProvider moduleProvider) {
      this.reactContext = reactContext;
      this.moduleInfoProvider = moduleInfoProvider;
      this.moduleProvider = moduleProvider;
    }

    @NonNull
    @Override
    public Iterator<ModuleHolder> iterator() {
      return new ModuleHolderIterator(reactContext, moduleInfoProvider, moduleProvider);
    }
  }

  /**
   * An interface for getting the Provider for a given module name.
   */
  private interface ModuleProvider {
    @Nullable Provider<NativeModule> get(String name, ReactApplicationContext reactContext);
  }

  /**
   * An implementation of Iterator that provides ModuleHolder objects for all the modules in the package.
   */
  private static class ModuleHolderIterator implements Iterator<ModuleHolder> {

    private final ReactApplicationContext reactContext;
    private final ReactModuleInfoProvider moduleInfoProvider;
    private final ModuleProvider moduleProvider;

    private final Iterator<Map.Entry<String, ReactModuleInfo>> entrySetIterator;
    private Map.Entry<String, ReactModuleInfo> nextEntry;

    private ModuleHolderIterator(
        ReactApplicationContext reactContext,
        ReactModuleInfoProvider moduleInfoProvider,
        ModuleProvider moduleProvider) {
      this.reactContext = reactContext;
      this.moduleInfoProvider = moduleInfoProvider;
      this.moduleProvider = moduleProvider;
      this.entrySetIterator = getEntrySetIterator(moduleInfoProvider);
      this.nextEntry = findNext();
    }

    @Override
    public boolean hasNext() {
      return nextEntry != null;
    }

    @Override
    public ModuleHolder next() {
      if (!hasNext()) {
        throw new NoSuchElementException("ModuleHolder not found");
      }

      Map.Entry<String, ReactModuleInfo> entry = nextEntry;
      nextEntry = findNext();

      String name = entry.getKey();
      ReactModuleInfo info = entry.getValue();
      Provider<NativeModule> provider = moduleProvider.get(name, reactContext);
      return new ModuleHolder(info, provider::get);
    }

    /**
     * Returns an iterator over all the ReactModuleInfo objects in the package.
     */
    private static Iterator<Map.Entry<String, ReactModuleInfo>> getEntrySetIterator(
        final ReactModuleInfoProvider moduleInfoProvider) {
      final Set<Map.Entry<String, ReactModuleInfo>> entrySet =
          moduleInfoProvider.getReactModuleInfos().entrySet();
      return entrySet.iterator();
    }

    /**
     * Finds the next module that should be returned by the iterator.
     * This skips over any TurboModules if TurboModules are enabled.
     */
    private Map.Entry<String, ReactModuleInfo> findNext() {
      while (entrySetIterator.hasNext()) {
        Map.Entry<String, ReactModuleInfo> entry = entrySetIterator.next();
        ReactModuleInfo info = entry.getValue();
        if (ReactFeatureFlags.useTurboModules && info.isTurboModule()) {
          continue;
        }
        return entry;
      }
      return null;
    }
  }

  /**
   * A base class for React Native packages that create View Managers.
   * Provides a default implementation for getViewManagerSpecs that uses the
   * reflection-based findViewManagerNameForReactClassName method.
   */
  public abstract static class ViewManagersPackage extends TurboModulesPackage {

    /**
     * Returns a list of module specs for the View Managers provided by this package.
     */
    @Override
    protected List<ModuleSpec> getViewManagerSpecs(ReactApplicationContext reactContext) {
      List<String> viewManagerNames = getViewManagerNames();
      if (viewManagerNames == null || viewManagerNames.isEmpty()) {
        return Collections.emptyList();
      }

      List<ModuleSpec> specs = new ArrayList<>();
      for (String name : viewManagerNames) {
        specs.add(
            new ModuleSpec(ViewManager.class, () -> createViewManager(reactContext, name), name));
      }
      return specs;
    }

    /**
     * Returns the list of fully qualified class names for the View Managers provided by this package.
     * If no View Managers are provided, returns null.
     */
    protected abstract List<String> getViewManagerNames();

    /**
     * Given a class name, creates a new instance of the corresponding View Manager.
     */
    protected ViewManager createViewManager(
        ReactApplicationContext reactContext, String viewManagerClassName) {
      try {
        Class<? extends ViewManager> viewManagerClass =
            Class.forName(viewManagerClassName).asSubclass(ViewManager.class);
        return viewManagerClass.getDeclaredConstructor().newInstance();
      } catch (Exception e) {
        throw new IllegalArgumentException(
            "Unable to instantiate ViewManager " + viewManagerClassName, e);
      }
    }
  }
} 