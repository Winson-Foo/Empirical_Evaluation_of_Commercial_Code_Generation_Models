package org.elasticsearch.plugins;

import org.elasticsearch.action.admin.cluster.node.info.PluginsAndModules;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.env.Environment;
import org.elasticsearch.env.TestEnvironment;
import org.elasticsearch.test.ESTestCase;
import org.junit.Before;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.MatcherAssert.assertThat;

/**
 * Unit tests for the MockPluginsService class
 */
public class MockPluginsServiceTests extends ESTestCase {

    /**
     * Test plugin 1
     */
    public static class TestPlugin1 extends Plugin {

        public TestPlugin1() {};

        // for map/flatmap/foreach testing
        @Override
        public List<String> getSettingsFilter() {
            return List.of("test value 1");
        }
    }

    /**
     * Test plugin 2
     */
    public static class TestPlugin2 extends Plugin {

        public TestPlugin2() {};

        // for map/flatmap/foreach testing
        @Override
        public List<String> getSettingsFilter() {
            return List.of("test value 2");
        }
    }

    private MockPluginsService mockPluginsService;

    /**
     * Runs before each test case
     */
    @Before
    public void setup() {
        List<Class<? extends Plugin>> classpathPlugins = List.of(TestPlugin1.class, TestPlugin2.class);
        Settings pathHomeSetting = Settings.builder().put(Environment.PATH_HOME_SETTING.getKey(), createTempDir()).build();
        this.mockPluginsService = new MockPluginsService(
            pathHomeSetting,
            TestEnvironment.newEnvironment(pathHomeSetting),
            classpathPlugins
        );
    }

    /**
     * Test superclass methods
     */
    public void testSuperclassMethods() {
        // Test map() method
        List<List<String>> mapResult = executeAndGetResult(mockPluginsService::map, Plugin::getSettingsFilter);
        assertThat(mapResult, containsInAnyOrder(List.of("test value 1"), List.of("test value 2")));

        // Test flatMap() method
        List<String> flatMapResult = executeAndGetResult(mockPluginsService::flatMap, Plugin::getSettingsFilter);
        assertThat(flatMapResult, containsInAnyOrder("test value 1", "test value 2"));

        // Test forEach() method
        List<String> forEachCollector = new ArrayList<>();
        mockPluginsService.forEach(p -> forEachCollector.addAll(p.getSettingsFilter()));
        assertThat(forEachCollector, containsInAnyOrder("test value 1", "test value 2"));

        // Test pluginMap() method
        Map<String, Plugin> pluginMap = mockPluginsService.pluginMap();
        assertThat(pluginMap.keySet(), containsInAnyOrder(containsString("TestPlugin1"), containsString("TestPlugin2")));

        // Test filterPlugins() method
        List<TestPlugin1> plugin1 = mockPluginsService.filterPlugins(TestPlugin1.class);
        assertThat(plugin1, contains(instanceOf(TestPlugin1.class)));
    }

    /**
     * Test info method
     */
    public void testInfo() {
        PluginsAndModules pluginsAndModulesInfo = this.mockPluginsService.info();

        // Test moduleInfos() method
        assertThat(pluginsAndModulesInfo.getModuleInfos(), empty());

        // Test pluginInfos() method
        List<String> pluginNames = pluginsAndModulesInfo.getPluginInfos().stream().map(PluginRuntimeInfo::descriptor).map(PluginDescriptor::getName).toList();
        assertThat(pluginNames, containsInAnyOrder(containsString("TestPlugin1"), containsString("TestPlugin2")));
    }

    /**
     * Method to execute a method with a function parameter and return its result
     * @param function The function to be executed
     * @param mapper The mapper function to get the result from the function
     * @param <T> The type of the result
     * @return The result of executing the function
     */
    private <T> T executeAndGetResult(Function<Function<Plugin, T>, List<T>> function, Function<Plugin, T> mapper) {
        return function.apply(mapper).stream().findFirst().orElseThrow();
    }
}