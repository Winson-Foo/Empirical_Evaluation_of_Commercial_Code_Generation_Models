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
import static org.hamcrest.MatcherAssert.assertThat;
import static org.mockito.Mockito.mock;

public class ForEachProcessorFactoryTests extends ESTestCase {

    private ScriptService scriptService;
    private Processor.Factory processorFactory;
    private ForEachProcessor.Factory forEachFactory;

    @Before
    public void setUp() {
        scriptService = mock(ScriptService.class);
        processorFactory = (r, t, description, c) -> new TestProcessor(ingestDocument -> {});
        forEachFactory = new ForEachProcessor.Factory(scriptService);
    }

    @Test
    public void testCreate() throws Exception {
        Map<String, Processor.Factory> registry = new HashMap<>();
        registry.put("_name", processorFactory);

        Map<String, Object> config = new HashMap<>();
        config.put("field", "_field");
        config.put("processor", Map.of("_name", Collections.emptyMap()));
        ForEachProcessor forEachProcessor = forEachFactory.create(registry, null, null, config);

        assertThat(forEachProcessor, notNullValue());
        assertThat(forEachProcessor.getField(), equalTo("_field"));
        assertThat(forEachProcessor.getInnerProcessor(), instanceOf(TestProcessor.class));
        assertFalse(forEachProcessor.isIgnoreMissing());
    }

    @Test
    public void testSetIgnoreMissing() throws Exception {
        Map<String, Processor.Factory> registry = new HashMap<>();
        registry.put("_name", processorFactory);

        Map<String, Object> config = new HashMap<>();
        config.put("field", "_field");
        config.put("processor", Map.of("_name", Collections.emptyMap()));
        config.put("ignore_missing", true);
        ForEachProcessor forEachProcessor = forEachFactory.create(registry, null, null, config);

        assertThat(forEachProcessor, notNullValue());
        assertThat(forEachProcessor.getField(), equalTo("_field"));
        assertThat(forEachProcessor.getInnerProcessor(), instanceOf(TestProcessor.class));
        assertTrue(forEachProcessor.isIgnoreMissing());
    }

    @Test
    public void testCreateWithTooManyProcessorTypes() throws Exception {
        Map<String, Processor.Factory> registry = new HashMap<>();
        registry.put("_first", processorFactory);
        registry.put("_second", processorFactory);

        Map<String, Object> config = new HashMap<>();
        config.put("field", "_field");
        Map<String, Object> processorTypes = new HashMap<>();
        processorTypes.put("_first", Map.of());
        processorTypes.put("_second", Map.of());
        config.put("processor", processorTypes);
        Exception exception = expectThrows(ElasticsearchParseException.class, () -> forEachFactory.create(registry, null, null, config));
        assertThat(exception.getMessage(), equalTo("[processor] Must specify exactly one processor type"));
    }

    @Test
    public void testCreateWithNonExistingProcessorType() throws Exception {
        Map<String, Object> config = new HashMap<>();
        config.put("field", "_field");
        config.put("processor", Map.of("_name", Collections.emptyMap()));
        Exception expectedException = expectThrows(
                ElasticsearchParseException.class,
                () -> forEachFactory.create(Map.of(), null, null, config)
        );
        assertThat(expectedException.getMessage(), equalTo("No processor type exists with name [_name]"));
    }

    @Test
    public void testCreateWithMissingField() {
        Map<String, Processor.Factory> registry = new HashMap<>();
        registry.put("_name", processorFactory);

        Map<String, Object> config = new HashMap<>();
        config.put("processor", List.of(Map.of("_name", Map.of())));

        Exception exception = expectThrows(Exception.class, () -> forEachFactory.create(registry, null, null, config));
        assertThat(exception.getMessage(), equalTo("[field] required property is missing"));
    }

    @Test
    public void testCreateWithMissingProcessor() {
        Map<String, Object> config = new HashMap<>();
        config.put("field", "_field");

        Exception exception = expectThrows(Exception.class, () -> forEachFactory.create(Map.of(), null, null, config));
        assertThat(exception.getMessage(), equalTo("[processor] required property is missing"));
    }
}