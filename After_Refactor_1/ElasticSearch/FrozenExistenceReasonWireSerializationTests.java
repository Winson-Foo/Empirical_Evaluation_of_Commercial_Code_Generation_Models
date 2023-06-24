package org.elasticsearch.xpack.autoscaling.existence;

import org.apache.commons.lang3.RandomStringUtils;
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.test.AbstractWireSerializingTestCase;

import java.util.ArrayList;
import java.util.List;

public class FrozenExistenceReasonWireSerializationTests extends AbstractWireSerializingTestCase<
        FrozenExistenceReason> {

    @Override
    protected Writeable.Reader<FrozenExistenceReason> instanceReader() {
        return FrozenExistenceReason::new;
    }

    @Override
    protected FrozenExistenceReason createTestInstance() {
        List<String> indices = FrozenExistenceReason.generateRandomIndices(between(0, 10));
        return new FrozenExistenceReason(indices);
    }

    /**
     * Mutates the given FrozenExistenceReason instance by either adding a new index to the list, or removing an existing
     * one from the list.
     *
     * @param instance the instance to mutate
     * @return the mutated instance
     */
    @Override
    protected FrozenExistenceReason mutateInstance(FrozenExistenceReason instance) {
        List<String> indices = new ArrayList<>(instance.indices());
        if (indices.isEmpty() || randomBoolean()) {
            indices.add(RandomStringUtils.randomAlphabetic(5));
        } else {
            indices.remove(between(0, indices.size() - 1));
        }
        return new FrozenExistenceReason(indices);
    }
}