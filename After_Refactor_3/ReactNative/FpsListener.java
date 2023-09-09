package com.facebook.react.views.scroll;

public interface FpsListener {

  void enable(String tag);

  void disable(String tag);

  boolean isEnabled();
} 