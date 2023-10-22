package org.elasticsearch.cluster.metadata;

import org.elasticsearch.common.io.stream.NamedWriteableRegistry;
import org.elasticsearch.common.util.Maps;
import org.elasticsearch.test.AbstractNamedWriteableTestCase;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class ComponentTemplateMetadataTests extends AbstractNamedWriteableTestCase<ComponentTemplateMetadata> {

    private static final String TEMPLATE_NAME = "test";
    private static final String NEW_TEMPLATE_NAME = "new";
    private static final String TEMPLATE_PATTERN = "pattern";
    private static final String NEW_TEMPLATE_PATTERN = "new_pattern";

    @Override
    protected ComponentTemplateMetadata createTestInstance() {
        Map<String, ComponentTemplate> templateMap = randomTemplateMap();
        return new ComponentTemplateMetadata(templateMap);
    }

    private Map<String, ComponentTemplate> randomTemplateMap() {
        int count = randomIntBetween(0, 3);
        Map<String, ComponentTemplate> templateMap = Maps.newMapWithExpectedSize(count);
        for (int i = 0; i < count; i++) {
            templateMap.put(randomAlphaOfLength(4), ComponentTemplateTests.randomInstance());
        }
        return templateMap;
    }

    @Override
    protected ComponentTemplateMetadata mutateInstance(ComponentTemplateMetadata instance) {
        if (instance.componentTemplates().size() == 0) {
            // Not really much to mutate, so just generate a new one
            return randomValueOtherThan(instance, this::createTestInstance);
        }
        Map<String, ComponentTemplate> templates = new HashMap<>(instance.componentTemplates());
        Map.Entry<String, ComponentTemplate> newTemplate = instance.componentTemplates().entrySet().iterator().next();
        if (randomBoolean()) {
            // Change the key
            templates.put(randomAlphaOfLength(4), newTemplate.getValue());
        } else {
            // Change the value
            templates.put(newTemplate.getKey(), ComponentTemplateTests.mutateTemplate(newTemplate.getValue()));
        }
        return new ComponentTemplateMetadata(templates);
    }

    @Override
    protected NamedWriteableRegistry getNamedWriteableRegistry() {
        return new NamedWriteableRegistry(
            Collections.singletonList(
                new NamedWriteableRegistry.Entry(
                    ComponentTemplateMetadata.class,
                    ComponentTemplateMetadata.TYPE,
                    ComponentTemplateMetadata::new
                )
            )
        );
    }

    @Override
    protected Class<ComponentTemplateMetadata> categoryClass() {
        return ComponentTemplateMetadata.class;
    }

    public void testComponentTemplateMetadata() {
        String pattern = "index*";
        Map<String, ComponentTemplate> templateMap = new HashMap<>();
        ComponentTemplate template = new ComponentTemplate(TEMPLATE_NAME, pattern, null, null, null, null);
        templateMap.put(TEMPLATE_PATTERN, template);
        ComponentTemplateMetadata componentTemplateMetadata = new ComponentTemplateMetadata(templateMap);
        assertEquals("Type must match", ComponentTemplateMetadata.TYPE, componentTemplateMetadata.type());
        assertTrue("Component templates must match", componentTemplateMetadata.componentTemplates().containsKey(TEMPLATE_PATTERN));
        assertEquals("Component template name must match", TEMPLATE_NAME, componentTemplateMetadata.componentTemplates().get(TEMPLATE_PATTERN).name());
        assertEquals("Component template pattern must match", pattern, componentTemplateMetadata.componentTemplates().get(TEMPLATE_PATTERN).patterns().get(0));
    }

    public void testComponentTemplateMetadataSerialization() throws Exception {
        Map<String, ComponentTemplate> templateMap = new HashMap<>();
        ComponentTemplate template = new ComponentTemplate(TEMPLATE_NAME, TEMPLATE_PATTERN, null, null, null, null);
        templateMap.put(TEMPLATE_PATTERN, template);
        ComponentTemplateMetadata componentTemplateMetadata = new ComponentTemplateMetadata(templateMap);
        ComponentTemplateMetadata newInstance = serializeAndDeserialize(componentTemplateMetadata);
        assertEquals("Component templates must match", componentTemplateMetadata.componentTemplates(), newInstance.componentTemplates());
    }

    public void testMutateComponentTemplateMetadata() throws Exception {
        Map<String, ComponentTemplate> templateMap = new HashMap<>();
        ComponentTemplate template = new ComponentTemplate(TEMPLATE_NAME, TEMPLATE_PATTERN, null, null, null, null);
        templateMap.put(TEMPLATE_PATTERN, template);
        ComponentTemplateMetadata componentTemplateMetadata = new ComponentTemplateMetadata(templateMap);

        // mutate the component template metadata
        ComponentTemplateMetadata mutatedComponentTemplateMetadata = mutateInstance(componentTemplateMetadata);

        // check that the old and new maps have the same keys
        assertEquals(componentTemplateMetadata.componentTemplates().keySet(), mutatedComponentTemplateMetadata.componentTemplates().keySet());

        // check that the value of the mutated key is different, or that a different key has been added
        assertFalse(mutatedComponentTemplateMetadata.componentTemplates().get(TEMPLATE_PATTERN).name().equals(TEMPLATE_NAME) ||
                    mutatedComponentTemplateMetadata.componentTemplates().containsKey(NEW_TEMPLATE_NAME));
    }
}