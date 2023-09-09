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

import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.instanceOf;

public class MockPluginsServiceTests extends ESTestCase {

    private static final String TEST_VALUE_1 = "test value 1";
    private static final String TEST_VALUE_2 = "test value 2";

    private MockPluginsService mockPluginsService;

    @Before
    public void setup() {
        List<Class<? extends Plugin>> classpathPlugins = List.of(TestPlugin1.class, TestPlugin2.class);
        Settings pathHomeSetting = Settings.builder().put(Environment.PATH_HOME_SETTING.getKey(), createTempDir()).build();
        Environment environment = TestEnvironment.newEnvironment(pathHomeSetting);
        this.mockPluginsService = new MockPluginsService(pathHomeSetting, environment, classpathPlugins);
    }

    public void testSuperclassMethods() {
        List<List<String>> mapResult = mockPluginsService.map(Plugin::getSettingsFilter).toList();
        assertThat(mapResult, containsInAnyOrder(List.of(TEST_VALUE_1), List.of(TEST_VALUE_2)));

        List<String> flatMapResult = mockPluginsService.flatMap(Plugin::getSettingsFilter).toList();
        assertThat(flatMapResult, containsInAnyOrder(TEST_VALUE_1, TEST_VALUE_2));

        List<String> forEachCollector = new ArrayList<>();
        mockPluginsService.forEach(p -> forEachCollector.addAll(p.getSettingsFilter()));
        assertThat(forEachCollector, containsInAnyOrder(TEST_VALUE_1, TEST_VALUE_2));

        Map<String, Plugin> pluginMap = mockPluginsService.pluginMap();
        assertThat(pluginMap.keySet(), containsInAnyOrder(containsString(TestPlugin1.class.getSimpleName()),
                containsString(TestPlugin2.class.getSimpleName())));

        List<TestPlugin1> plugin1 = mockPluginsService.filterPlugins(TestPlugin1.class);
        assertThat(plugin1, contains(instanceOf(TestPlugin1.class)));
    }

    public void testInfo() {
        PluginsAndModules pam = this.mockPluginsService.info();

        assertThat(pam.getModuleInfos(), empty());

        List<String> pluginNames = pam.getPluginInfos().stream().map(PluginRuntimeInfo::descriptor).map(PluginDescriptor::getName).toList();
        assertThat(pluginNames, containsInAnyOrder(containsString(TestPlugin1.class.getSimpleName()),
                containsString(TestPlugin2.class.getSimpleName())));
    }

    static class TestPlugin1 extends Plugin implements TestPluginInterface {
        public TestPlugin1() {};

        @Override
        public List<String> getSettingsFilter() {
            return List.of(TEST_VALUE_1);
        }
    }

    static class TestPlugin2 extends Plugin implements TestPluginInterface {
        public TestPlugin2() {};

        @Override
        public List<String> getSettingsFilter() {
            return List.of(TEST_VALUE_2);
        }
    }

    interface TestPluginInterface {
        List<String> getSettingsFilter();
    }
} 