/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 *
 */

package org.elasticsearch.xpack.core.ilm.action;

import org.elasticsearch.action.support.IndicesOptions;
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.test.AbstractWireSerializingTestCase;
import org.elasticsearch.xpack.core.ilm.action.RetryAction.Request;

import java.util.Arrays;

public class RetryRequestTests extends AbstractWireSerializingTestCase<Request> {

    /**
     * Creates a test instance of the RetryRequest, with optional randomization of indices and indicesOptions.
     *
     * @return a Request instance
     */
    @Override
    protected Request createTestInstance() {
        Request request = new Request();
        if (randomBoolean()) {
            request.indices(generateRandomStringArray(20, 20, false));
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

    /**
     * Creates a new instance of Request using the provided reader object.
     *
     * @return a Reader instance of Request
     */
    @Override
    protected Writeable.Reader<Request> instanceReader() {
        return Request::new;
    }

    /**
     * Mutates the instance of Request by randomly changing either its indices or indicesOptions field.
     *
     * @param instance the instance of Request to be mutated
     * @return a mutated instance of Request
     */
    @Override
    protected Request mutateInstance(Request instance) {
        String[] indices = instance.indices();
        IndicesOptions indicesOptions = instance.indicesOptions();
        
        int mutationType = between(0, 1);
        if (mutationType == 0) {
            indices = randomValueOtherThanMany(
                i -> Arrays.equals(i, instance.indices()),
                () -> generateRandomStringArray(20, 10, false, true)
            );
        } else if (mutationType == 1) {
            indicesOptions = randomValueOtherThan(
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
        } else {
            throw new AssertionError("Illegal randomisation branch");
        }
        
        Request newRequest = new Request();
        newRequest.indices(indices);
        newRequest.indicesOptions(indicesOptions);
        return newRequest;
    }
}