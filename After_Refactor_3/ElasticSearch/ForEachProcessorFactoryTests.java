package org.elasticsearch.ingest.common;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.elasticsearch.ElasticsearchParseException;
import org.elasticsearch.ingest.Processor;
import org.elasticsearch.ingest.TestProcessor;
import org.elasticsearch.script.ScriptService;
import org.elasticsearch.test.ESTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;

import static org.hamcrest.Matchers.*;
import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class ForEachProcessorFactoryTests extends ESTestCase {

    private static final String FIELD = "_field";
    private static final Map<String, Object> EMPTY_PROCESSOR = Collections.emptyMap();
    private static final String PROCESSOR_TYPE = "_name";

    @Mock
    private Processor.Factory processorFactory;

    private ForEachProcessor.Factory forEachFactory;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        forEachFactory = new ForEachProcessor.Factory(mock(ScriptService.class));
    }

    @Test
    public void testCreate() throws Exception {
        Processor expectedProcessor = new TestProcessor(ingestDocument -> {});
        Map<String, Processor.Factory> registry = Collections.singletonMap(PROCESSOR_TYPE, processorFactory);
        when(processorFactory.create(eq(registry), eq(null), eq(null), eq(EMPTY_PROCESSOR)))
            .thenReturn(expectedProcessor);

        Map<String, Object> config = createConfig(List.of(EMPTY_PROCESSOR));
        ForEachProcessor processor = forEachFactory.create(registry, null, null, config);

        assertThat(processor, is(notNullValue()));
        assertThat(processor.getField(), is(equalTo(FIELD)));
        assertThat(processor.getInnerProcessor(), is(sameInstance(expectedProcessor)));
        assertThat(processor.isIgnoreMissing(), is(false));
    }

    @Test
    public void testCreateWithMissingField() throws Exception {
        Map<String, Processor.Factory> registry = Collections.singletonMap(PROCESSOR_TYPE, processorFactory);
        when(processorFactory.create(eq(registry), eq(null), eq(null), eq(EMPTY_PROCESSOR)))
            .thenReturn(new TestProcessor(ingestDocument -> {}));

        Map<String, Object> config = new HashMap<>();
        config.put("processor", List.of(Map.of(PROCESSOR_TYPE, EMPTY_PROCESSOR)));
        Exception exception = expectThrows(ElasticsearchParseException.class, 
            () -> forEachFactory.create(registry, null, null, config));
        assertThat(exception.getMessage(), is(equalTo("[field] required property is missing")));
    }

    @Test
    public void testCreateWithMissingProcessor() {
        Map<String, Processor.Factory> registry = Collections.singletonMap(PROCESSOR_TYPE, processorFactory);

        Map<String, Object> config = new HashMap<>();
        config.put("field", FIELD);
        Exception exception = expectThrows(Exception.class, 
            () -> forEachFactory.create(registry, null, null, config));
        assertThat(exception.getMessage(), is(equalTo("[processor] required property is missing")));
    }

    @Test
    public void testCreateWithNonExistingProcessorType() throws Exception {
        Map<String, Object> config = createConfig(List.of(Map.of(PROCESSOR_TYPE, EMPTY_PROCESSOR)));
        Exception expectedException = expectThrows(ElasticsearchParseException.class,
            () -> forEachFactory.create(new HashMap<>(), null, null, config));
        assertThat(expectedException.getMessage(), is(equalTo("No processor type exists with name [" + PROCESSOR_TYPE + "]")));
    }

    @Test
    public void testCreateWithTooManyProcessorTypes() throws Exception {
        String secondProcessorType = "_second";
        Map<String, Processor.Factory> registry = new HashMap<>();
        registry.put(PROCESSOR_TYPE, processorFactory);
        registry.put(secondProcessorType, processorFactory);

        Map<String, Object> config = createConfig(Map.of(PROCESSOR_TYPE, EMPTY_PROCESSOR, secondProcessorType, EMPTY_PROCESSOR));
        Exception exception = expectThrows(ElasticsearchParseException.class, 
            () -> forEachFactory.create(registry, null, null, config));
        assertThat(exception.getMessage(), is(equalTo("[processor] Must specify exactly one processor type")));
    }

    @Test
    public void testSetIgnoreMissing() throws Exception {
        Processor expectedProcessor = new TestProcessor(ingestDocument -> {});
        Map<String, Processor.Factory> registry = Collections.singletonMap(PROCESSOR_TYPE, processorFactory);
        when(processorFactory.create(eq(registry), eq(null), eq(null), eq(EMPTY_PROCESSOR)))
            .thenReturn(expectedProcessor);

        Map<String, Object> config = createConfig(List.of(EMPTY_PROCESSOR), true);
        ForEachProcessor processor = forEachFactory.create(registry, null, null, config);

        assertThat(processor, is(notNullValue()));
        assertThat(processor.getField(), is(equalTo(FIELD)));
        assertThat(processor.getInnerProcessor(), is(sameInstance(expectedProcessor)));
        assertThat(processor.isIgnoreMissing(), is(true));
    }

    private Map<String, Object> createConfig(List<Map<String, Object>> processors) {
        return createConfig(processors, false);
    }

    private Map<String, Object> createConfig(List<Map<String, Object>> processors, boolean ignoreMissing) {
        Map<String, Object> config = new HashMap<>();
        config.put("field", FIELD);
        config.put("processor", processors);
        config.put("ignore_missing", ignoreMissing);
        return config;
    }
}