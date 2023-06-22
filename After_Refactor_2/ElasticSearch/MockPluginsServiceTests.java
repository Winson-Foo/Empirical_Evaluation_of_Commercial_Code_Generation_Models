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

import static org.hamcrest.Matchers.*;
import static org.junit.Assert.assertThat;

public class MockPluginsServiceTests extends ESTestCase {
    private MockPluginsService mockPluginsService;
    private List<Class<? extends Plugin>> classpathPlugins;

    private interface PluginSettingsFilter {
        List<String> getSettingsFilter();
    }

    private class TestPlugin1 extends Plugin implements PluginSettingsFilter {
        @Override
        public List<String> getSettingsFilter() {
            return List.of("test value 1");
        }
    }

    private class TestPlugin2 extends Plugin implements PluginSettingsFilter {
        @Override
        public List<String> getSettingsFilter() {
            return List.of("test value 2");
        }
    }

    @Before
    public void setUp() {
        classpathPlugins = List.of(TestPlugin1.class, TestPlugin2.class);
        Settings pathHomeSetting = Settings.builder().put(Environment.PATH_HOME_SETTING.getKey(), createTempDir()).build();
        mockPluginsService = new MockPluginsService(
                pathHomeSetting,
                TestEnvironment.newEnvironment(pathHomeSetting),
                classpathPlugins
        );
    }

    public void testGetSettingsFilter() {
        List<List<String>> pluginSettings = mockPluginsService.map(PluginSettingsFilter::getSettingsFilter).toList();
        assertThat(pluginSettings, containsInAnyOrder(List.of("test value 1"), List.of("test value 2")));

        List<String> allSettings = mockPluginsService.flatMap(PluginSettingsFilter::getSettingsFilter).toList();
        assertThat(allSettings, containsInAnyOrder("test value 1", "test value 2"));

        List<String> allSettingsCollector = new ArrayList<>();
        mockPluginsService.forEach(p -> allSettingsCollector.addAll(p.getSettingsFilter()));
        assertThat(allSettingsCollector, containsInAnyOrder("test value 1", "test value 2"));

        Map<String, Plugin> pluginMap = mockPluginsService.pluginMap();
        assertThat(pluginMap.keySet(), containsInAnyOrder(containsString("TestPlugin1"), containsString("TestPlugin2")));

        List<TestPlugin1> plugin1 = mockPluginsService.filterPlugins(TestPlugin1.class);
        assertThat(plugin1, contains(instanceOf(TestPlugin1.class)));
    }

    public void testGetPluginInfo() {
        PluginsAndModules pluginsAndModules = mockPluginsService.info();

        assertThat(pluginsAndModules.getModuleInfos(), empty());

        List<String> pluginNames = pluginsAndModules.getPluginInfos().stream()
                .map(PluginRuntimeInfo::descriptor)
                .map(PluginDescriptor::getName)
                .toList();
        assertThat(pluginNames, containsInAnyOrder(containsString("TestPlugin1"), containsString("TestPlugin2")));
    }
}

