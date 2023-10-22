package org.elasticsearch.xpack.core.ilm.action;

import org.elasticsearch.action.support.IndicesOptions;
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.test.AbstractWireSerializingTestCase;
import org.elasticsearch.xpack.core.ilm.action.RetryAction.Request;

import java.util.Arrays;

public class RetryRequestTests extends AbstractWireSerializingTestCase<Request> {

    private static final int MIN_INDEX_LENGTH = 10;
    private static final int MAX_INDEX_LENGTH = 20;

    @Override
    protected Request createTestInstance() {
        Request request = new Request();
        if (randomBoolean()) {
            request.indices(generateRandomStrings());
        }
        if (randomBoolean()) {
            IndicesOptions indicesOptions = IndicesOptions.fromOptions(
                randomBoolean(),
                randomBoolean(),
                randomBoolean(),
                randomBoolean(),
                randomBoolean(),
                randomBoolean(),
                randomBoolean(),
                randomBoolean()
            );
            request.indicesOptions(indicesOptions);
        }
        return request;
    }

    @Override
    protected Writeable.Reader<Request> instanceReader() {
        return Request::new;
    }

    @Override
    protected Request mutateInstance(Request instance) {
        String[] indices = instance.indices();
        IndicesOptions indicesOptions = instance.indicesOptions();
        int branch = between(0, 1);
        switch (branch) {
            case 0 -> indices = generateRandomStrings();
            case 1 -> indicesOptions = generateIndicesOptions();
            default -> throw new AssertionError("Illegal randomisation branch");
        }
        Request newRequest = new Request();
        newRequest.indices(indices);
        newRequest.indicesOptions(indicesOptions);
        return newRequest;
    }
    
    private String[] generateRandomStrings() {
        return randomValueOtherThanMany(
                i -> Arrays.equals(i, instance.indices()),
                () -> generateRandomStringArray(MAX_INDEX_LENGTH, MIN_INDEX_LENGTH, false, true)
        );
    }

    private IndicesOptions generateIndicesOptions() {
        return randomValueOtherThan(
            indicesOptions,
            () -> IndicesOptions.fromOptions(
                randomBoolean(), 
                randomBoolean(),
                randomBoolean(),
                randomBoolean(),
                randomBoolean(),
                randomBoolean(),
                randomBoolean(),
                randomBoolean()
            )
        );
    }
}

