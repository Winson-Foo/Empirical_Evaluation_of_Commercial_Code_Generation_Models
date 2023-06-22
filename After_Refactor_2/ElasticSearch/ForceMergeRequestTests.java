package org.elasticsearch.action.admin.indices.forcemerge;

import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.not;
import static org.hamcrest.Matchers.notNullValue;
import static org.hamcrest.Matchers.nullValue;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;

import org.elasticsearch.action.ActionRequestValidationException;
import org.elasticsearch.test.ESTestCase;

public class ForceMergeRequestTests extends ESTestCase {

    private static final String[] INDICES = {"shop", "blog"};
    private static final int MAX_SEGMENTS = 12;
    private static final int DEFAULT_MAX_SEGMENTS = ForceMergeRequest.Defaults.MAX_NUM_SEGMENTS;
    private static final String DESCRIPTION_TEMPLATE = "Force-merge indices [%s], maxSegments[%d], onlyExpungeDeletes[%s], flush[%s]";

    @Test
    public void testValidateWithExpungeDeletesAndMaxSegments() {
        boolean flush = randomBoolean();
        boolean onlyExpungeDeletes = randomBoolean();
        int maxNumSegments = randomIntBetween(DEFAULT_MAX_SEGMENTS, 100);

        ForceMergeRequest request = new ForceMergeRequest();
        request.flush(flush);
        request.onlyExpungeDeletes(onlyExpungeDeletes);
        request.maxNumSegments(maxNumSegments);

        assertThat(request.flush(), equalTo(flush));
        assertThat(request.onlyExpungeDeletes(), equalTo(onlyExpungeDeletes));
        assertThat(request.maxNumSegments(), equalTo(maxNumSegments));

        ActionRequestValidationException validation = request.validate();
        if (onlyExpungeDeletes && maxNumSegments != DEFAULT_MAX_SEGMENTS) {
            assertThat(validation, notNullValue());
            assertThat(validation.validationErrors(),
                    contains("cannot set only_expunge_deletes and max_num_segments at the same time, those two parameters are mutually exclusive"));
        } else {
            assertThat(validation, nullValue());
        }
    }

    @Test
    public void testValidateWithDefaultValues() {
        boolean flush = randomBoolean();
        boolean onlyExpungeDeletes = randomBoolean();
        int maxNumSegments = DEFAULT_MAX_SEGMENTS;

        ForceMergeRequest request = new ForceMergeRequest();
        request.flush(flush);
        request.onlyExpungeDeletes(onlyExpungeDeletes);
        request.maxNumSegments(maxNumSegments);

        assertThat(request.flush(), equalTo(flush));
        assertThat(request.onlyExpungeDeletes(), equalTo(onlyExpungeDeletes));
        assertThat(request.maxNumSegments(), equalTo(maxNumSegments));
        assertThat(request.validate(), nullValue());
    }

    @Test
    public void testDescriptionWithDefaultValues() {
        ForceMergeRequest request = new ForceMergeRequest();
        assertEquals(String.format(DESCRIPTION_TEMPLATE, "", -1, false, true), request.getDescription());
    }

    @Test
    public void testDescriptionWithIndices() {
        ForceMergeRequest request = new ForceMergeRequest(INDICES);
        assertEquals(String.format(DESCRIPTION_TEMPLATE, String.join(", ", INDICES), -1, false, true), request.getDescription());
    }

    @Test
    public void testDescriptionWithCustomValues() {
        ForceMergeRequest request = new ForceMergeRequest();
        request.maxNumSegments(MAX_SEGMENTS);
        request.onlyExpungeDeletes(true);
        request.flush(false);
        assertEquals(String.format(DESCRIPTION_TEMPLATE, "", MAX_SEGMENTS, true, false), request.getDescription());
    }

}