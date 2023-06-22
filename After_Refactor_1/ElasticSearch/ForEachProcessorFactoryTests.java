package org.elasticsearch.ingest.common;

import org.elasticsearch.ElasticsearchParseException;
import org.elasticsearch.ingest.Processor;
import org.elasticsearch.ingest.TestProcessor;
import org.elasticsearch.script.ScriptService;
import org.elasticsearch.test.ESTestCase;
import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.hamcrest.Matchers.*;
import static org.hamcrest.Matchers.sameInstance;
import static org.junit.Assert.*;

public class ForEachProcessorFactoryTests extends ESTestCase {

    private ScriptService scriptService;
    private Processor testProcessor;
    private ForEachProcessor.Factory forEachFactory;
    private Map<String, Processor.Factory> registry;

    @Before
    public void setUp() {
        scriptService = mock(ScriptService.class);
        testProcessor = new TestProcessor(ingestDocument -> {});
        forEachFactory = new ForEachProcessor.Factory(scriptService);
        registry = new HashMap<>();
        registry.put("_name", (r, t, description, c) -> testProcessor);
    }

    @Test
    public void testCreate() throws Exception {
        Map<String, Object> config = getValidConfig();
        ForEachProcessor forEachProcessor = forEachFactory.create(registry, null, null, config);

        assertThat(forEachProcessor, notNullValue());
        assertThat(forEachProcessor.getField(), equalTo("_field"));
        assertThat(forEachProcessor.getInnerProcessor(), sameInstance(testProcessor));
        assertFalse(forEachProcessor.isIgnoreMissing());
    }

    @Test
    public void testSetIgnoreMissing() throws Exception {
        Map<String, Object> config = getValidConfig();
        config.put("ignore_missing", true);
        ForEachProcessor forEachProcessor = forEachFactory.create(registry, null, null, config);

        assertThat(forEachProcessor, notNullValue());
        assertThat(forEachProcessor.getField(), equalTo("_field"));
        assertThat(forEachProcessor.getInnerProcessor(), sameInstance(testProcessor));
        assertTrue(forEachProcessor.isIgnoreMissing());
    }

    @Test(expected = ElasticsearchParseException.class)
    public void testCreateWithTooManyProcessorTypes() throws Exception {
        Map<String, Object> config = getValidConfig();
        config.put("processor", Map.of("_first", Map.of(), "_second", Map.of()));
        forEachFactory.create(registry, null, null, config);
    }

    @Test(expected = ElasticsearchParseException.class)
    public void testCreateWithNonExistingProcessorType() throws Exception {
        Map<String, Object> config = getValidConfig();
        config.put("processor", Map.of("_invalid", Map.of()));
        forEachFactory.create(new HashMap<>(), null, null, config);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testCreateWithMissingField() throws Exception {
        Map<String, Object> config = new HashMap<>();
        config.put("processor", Collections.singletonList(Map.of("_name", Map.of())));
        forEachFactory.create(registry, null, null, config);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testCreateWithMissingProcessor() {
        Map<String, Object> config = new HashMap<>();
        config.put("field", "_field");
        forEachFactory.create(new HashMap<>(), null, null, config);
    }

    private Map<String, Object> getValidConfig() {
        Map<String, Object> processorConfig = Collections.singletonMap("_name", Collections.emptyMap());
        Map<String, Object> config = new HashMap<>();
        config.put("field", "_field");
        config.put("processor", processorConfig);
        return config;
    }
}