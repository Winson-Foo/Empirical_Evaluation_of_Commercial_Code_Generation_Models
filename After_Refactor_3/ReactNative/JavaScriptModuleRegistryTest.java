private void assertJSModuleName(Class<?> moduleClass, String expectedName) {
    String name = JavaScriptModuleRegistry.getJSModuleName(moduleClass);
    Assert.assertEquals(expectedName, name);
}

@Test
public void testGetJSModuleName() {
    assertJSModuleName(TestJavaScriptModule.class, "TestJavaScriptModule");
}

@Test
public void testGetJSModuleName_stripOuterClass() {
    assertJSModuleName(OuterClass$NestedInnerClass.class, "NestedInnerClass");
}

@Test
public void testGetJSModuleName_emptyName() {
    class EmptyName implements JavaScriptModule {
        void doSomething();
    }

    assertJSModuleName(EmptyName.class, "");
}

@Test
public void testGetJSModuleName_nullArgument() {
    assertJSModuleName(null, "");
}

@Test
public void testGetJSModuleName_innerClass() {
    class OuterClass {
        interface InnerClass extends JavaScriptModule {
            void doSomething();
        }
    }

    assertJSModuleName(OuterClass.InnerClass.class, "InnerClass");
}

@Test
public void testGetJSModuleName_anonymousClass() {
    JavaScriptModule anonymousModule = new JavaScriptModule() {
        void doSomething();
    };

    assertJSModuleName(anonymousModule.getClass(), "");
}