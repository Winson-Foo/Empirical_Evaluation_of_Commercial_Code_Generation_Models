package org.elasticsearch.xpack.core.action.util;

import java.io.IOException;
import java.util.List;
import java.util.Objects;

import org.elasticsearch.ResourceNotFoundException;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.xcontent.ParseField;
import org.elasticsearch.xcontent.ToXContent;
import org.elasticsearch.xcontent.ToXContentObject;
import org.elasticsearch.xcontent.XContentBuilder;

/**
 * Generic wrapper class for a page of query results and the total number of
 * query results.<br>
 * {@linkplain #count()} is the total number of results but that value may
 * not be equal to the actual length of the {@linkplain #results()} list if from
 * &amp; take or some cursor was used in the database query.
 */
public final class QueryPage<T extends ToXContent & Writeable> implements ToXContentObject, Writeable {

    private static final String COULD_NOT_FIND_REQUESTED = "Could not find requested ";
    private static final String RESULTS_FIELD = "results_field";
    private static final String COUNT_NAME = "count";

    private final ParseField resultsField;
    private final List<T> results;
    private final long count;

    public QueryPage(List<T> results, long count, ParseField resultsField) {
        this.results = results;
        this.count = count;
        this.resultsField = Objects.requireNonNull(resultsField);
    }

    public QueryPage(StreamInput in, Reader<T> hitReader) throws IOException {
        resultsField = new ParseField(in.readString());
        results = in.readList(hitReader);
        count = in.readLong();
    }

    /**
     * Get the exception for an empty query page.
     *
     * @param resultsField the parse field for the results
     * @return the exception for an empty query page
     */
    public static ResourceNotFoundException emptyQueryPage(ParseField resultsField) {
        return new ResourceNotFoundException(COULD_NOT_FIND_REQUESTED + resultsField.getPreferredName());
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(resultsField.getPreferredName());
        out.writeList(results);
        out.writeLong(count);
    }

    /**
     * Serialize the query page to XContent.
     *
     * @param builder the XContent builder
     * @param params  the parameters for the XContent
     * @return the serialized XContent
     * @throws IOException if there is an error during serialization
     */
    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(COUNT_NAME, count);
        startResultsField(builder);
        serializeResults(builder, params);
        endResultsField(builder);
        builder.endObject();
        return builder;
    }

    /**
     * Start serializing the results array.
     *
     * @param builder the XContent builder
     * @throws IOException if there is an error during serialization
     */
    private void startResultsField(XContentBuilder builder) throws IOException {
        builder.startArray(resultsField.getPreferredName());
    }

    /**
     * Serialize the query page results to XContent.
     *
     * @param builder the XContent builder
     * @param params  the parameters for the XContent
     * @throws IOException if there is an error during serialization
     */
    private void serializeResults(XContentBuilder builder, Params params) throws IOException {
        for (T result : results) {
            if (result != null) {
                result.toXContent(builder, params);
            }
        }
    }

    /**
     * End serializing the results array.
     *
     * @param builder the XContent builder
     * @throws IOException if there is an error during serialization
     */
    private void endResultsField(XContentBuilder builder) throws IOException {
        builder.endArray();
    }

    public List<T> results() {
        return results;
    }

    public long count() {
        return count;
    }

    public ParseField getResultsField() {
        return resultsField;
    }

    @Override
    public int hashCode() {
        return Objects.hash(results, count);
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null) {
            return false;
        }

        if (getClass() != obj.getClass()) {
            return false;
        }

        @SuppressWarnings("unchecked")
        QueryPage<T> other = (QueryPage<T>) obj;
        return Objects.equals(results, other.results) && Objects.equals(count, other.count);
    }
}