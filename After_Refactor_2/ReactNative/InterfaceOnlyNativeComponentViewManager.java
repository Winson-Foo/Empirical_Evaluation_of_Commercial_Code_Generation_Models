public class InterfaceOnlyNativeComponentViewManager extends SimpleViewManager<ViewGroup>  {

  public static final String REACT_CLASS = "InterfaceOnlyNativeComponentView";

  @Override
  public String getName() {
    return REACT_CLASS;
  }

  @Override
  public ViewGroup createViewInstance(ThemedReactContext context) {
    throw new IllegalStateException();
  }

  @Override
  public void setTitle(ViewGroup view, String value) {}
}

public class InterfaceOnlyNativeComponentViewManagerDelegate implements
     InterfaceOnlyNativeComponentViewManagerInterface {

    private final InterfaceOnlyNativeComponentViewManager viewManager;

    InterfaceOnlyNativeComponentViewManagerDelegate(InterfaceOnlyNativeComponentViewManager viewManager) {
        this.viewManager = viewManager;
    }

    @Override
    public void setTitle(ViewGroup view, String title) {
        viewManager.setTitle(view, title);
    }
}

public interface InterfaceOnlyNativeComponentViewManagerInterface<T extends ViewGroup> {
    void setTitle(T view, String title);
}