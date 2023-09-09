package org.elasticsearch.cluster.metadata;

import org.elasticsearch.common.io.stream.NamedWriteableRegistry;
import org.elasticsearch.common.util.Maps;
import org.elasticsearch.test.AbstractNamedWriteableTestCase;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class ComponentTemplateMetadataTests extends AbstractNamedWriteableTestCase<ComponentTemplateMetadata> {

    private static final int MAX_TEMPLATE_COUNT = 3;
    private static final int TEMPLATE_NAME_LENGTH = 4;

    @Override
    protected ComponentTemplateMetadata createTestInstance() {
        int count = randomIntBetween(0, MAX_TEMPLATE_COUNT);
        Map<String, ComponentTemplate> templateMap = Maps.newMapWithExpectedSize(count);
        for (int i = 0; i < count; i++) {
            templateMap.put(randomAlphaOfLength(TEMPLATE_NAME_LENGTH), ComponentTemplateTests.createTestInstance());
        }
        return new ComponentTemplateMetadata(templateMap);
    }

    @Override
    protected ComponentTemplateMetadata mutateInstance(ComponentTemplateMetadata instance) {
        if (instance.componentTemplates().isEmpty()) {
            // Not really much to mutate, so just generate a new one
            return randomValueOtherThan(instance, this::createTestInstance);
        }
        Map<String, ComponentTemplate> templates = new HashMap<>(instance.componentTemplates());
        Map.Entry<String, ComponentTemplate> newTemplate = instance.componentTemplates().entrySet().iterator().next();
        if (randomBoolean()) {
            // Change the key
            templates.put(randomAlphaOfLength(TEMPLATE_NAME_LENGTH), newTemplate.getValue());
        } else {
            // Change the value
            templates.put(newTemplate.getKey(), ComponentTemplateTests.mutateInstance(newTemplate.getValue()));
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
}