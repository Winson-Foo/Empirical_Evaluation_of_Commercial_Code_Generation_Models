package com.facebook.react.bridge;

import java.util.List;

public interface JSIModuleInitializer {

  List<JSIModuleSpec> getJSIModules(ReactApplicationContext context, JavaScriptContextHolder holder);

  String CONTEXT_PARAM = "reactApplicationContext";
  String HOLDER_PARAM = "javascriptContextHolder";
} 