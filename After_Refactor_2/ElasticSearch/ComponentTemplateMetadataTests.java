package org.elasticsearch.cluster.metadata;

import org.elasticsearch.test.ESTestCase;

public class ComponentTemplateMetadataTests extends ESTestCase {

    public void testCreate() {
        int count = randomIntBetween(0, 3);
        ComponentTemplateMetadata metadata = ComponentTemplateMetadataFactory.create(count);
        assertEquals(count, metadata.componentTemplates().size());
    }

    public void testMutate() {
        ComponentTemplateMetadata metadata = ComponentTemplateMetadataFactory.create(randomIntBetween(1, 3));
        ComponentTemplateMetadata mutated = metadata.mutate(new ComponentTemplateMetadataMutator());
        assertNotSame(metadata, mutated);
    }

    public void testSerialization() throws IOException {
        ComponentTemplateMetadata metadata = ComponentTemplateMetadataFactory.create(randomIntBetween(1, 3));
        BytesStreamOutput out = new BytesStreamOutput();
        ComponentTemplateMetadataSerializer.serialize(metadata, out);
        StreamInput in = new NamedWriteableAwareInputStream(new BytesStreamInput(out.bytes()));
        ComponentTemplateMetadata deserialized = ComponentTemplateMetadataSerializer.deserialize(in);
        assertEquals(metadata.componentTemplates(), deserialized.componentTemplates());
    }
}