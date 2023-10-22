package org.elasticsearch.action.admin.indices.forcemerge;

import org.elasticsearch.action.ActionRequestValidationException;
import org.elasticsearch.test.ESTestCase;

import static org.hamcrest.Matchers.*;

public class ForceMergeRequestTests extends ESTestCase {

    public void testValidate() {
        boolean shouldFlush = randomBoolean();
        boolean shouldExpungeDeletes = randomBoolean();
        int maxSegments = randomIntBetween(ForceMergeRequest.Defaults.MAX_NUM_SEGMENTS, 100);

        ForceMergeRequest request = new ForceMergeRequest()
                .flush(shouldFlush)
                .onlyExpungeDeletes(shouldExpungeDeletes)
                .maxNumSegments(maxSegments);

        assertThat(request.shouldFlush(), equalTo(shouldFlush));
        assertThat(request.shouldExpungeDeletes(), equalTo(shouldExpungeDeletes));
        assertThat(request.maxNumSegments(), equalTo(maxSegments));

        ActionRequestValidationException validation = request.validate();
        if (shouldExpungeDeletes && maxSegments != ForceMergeRequest.Defaults.MAX_NUM_SEGMENTS) {
            assertThat(validation, notNullValue());
            assertThat(validation.validationErrors(), contains("cannot set only_expunge_deletes and max_num_segments at the same time, those two parameters are mutually exclusive"));
            return;
        }
        assertThat(validation, nullValue());
    }

    public void testDescription() {
        ForceMergeRequest request = new ForceMergeRequest()
                .indices()
                .maxNumSegments(-1)
                .onlyExpungeDeletes(false)
                .flush(true);
        assertEquals("Force-merge indices [], maxSegments[-1], onlyExpungeDeletes[false], flush[true]", request.getDescription());

        request = new ForceMergeRequest()
                .indices("shop", "blog")
                .maxNumSegments(-1)
                .onlyExpungeDeletes(false)
                .flush(true);
        assertEquals("Force-merge indices [shop, blog], maxSegments[-1], onlyExpungeDeletes[false], flush[true]", request.getDescription());

        request = new ForceMergeRequest()
                .indices()
                .maxNumSegments(12)
                .onlyExpungeDeletes(true)
                .flush(false);
        assertEquals("Force-merge indices [], maxSegments[12], onlyExpungeDeletes[true], flush[false]", request.getDescription());
    }
}