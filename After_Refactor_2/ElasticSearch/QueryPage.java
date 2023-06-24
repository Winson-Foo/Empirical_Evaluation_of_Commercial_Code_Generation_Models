package org.elasticsearch.xpack.core.action.util;

import org.elasticsearch.ResourceNotFoundException;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.xcontent.ParseField;
import org.elasticsearch.xcontent.ToXContent;
import org.elasticsearch.xcontent.ToXContentObject;
import org.elasticsearch.xcontent.XContentBuilder;

import java.io.IOException;
import java.util.List;
import java.util.Objects;

public final class QueryPage<T extends ToXContent & Writeable> implements ToXContentObject, Writeable {

    public static final ParseField COUNT = new ParseField("count");
    public static final ParseField RESULTS_FIELD = new ParseField("results");

    private final List<T> results;
    private final long count;
    private final ParseField resultsField;

    public QueryPage(List<T> results, long count) {
        this(results, count, RESULTS_FIELD);
    }

    public QueryPage(List<T> results, long count, ParseField resultsField) {
        this.results = Objects.requireNonNull(results);
        this.count = count;
        this.resultsField = Objects.requireNonNull(resultsField);
    }

    public QueryPage(StreamInput in, Reader<T> reader) throws IOException {
        this.resultsField = new ParseField(in.readString());
        this.results = in.readList(reader);
        this.count = in.readLong();
    }

    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        doXContentBody(builder, params);
        builder.endObject();
        return builder;
    }

    public XContentBuilder doXContentBody(XContentBuilder builder, Params params) throws IOException {
        builder.field(COUNT.getPreferredName(), count);
        builder.startArray(resultsField.getPreferredName());
        for (T result : results) {
            if (result != null) {
                result.toXContent(builder, params);
            }
        }
        builder.endArray();
        return builder;
    }

    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(resultsField.getPreferredName());
        out.writeList(results);
        out.writeLong(count);
    }

    public List<T> getResults() {
        return results;
    }

    public long getCount() {
        return count;
    }

    public ParseField getResultsField() {
        return resultsField;
    }

    public static ResourceNotFoundException empty() {
        return new ResourceNotFoundException("No results found");
    }

    @Override
    public int hashCode() {
        return Objects.hash(results, count, resultsField);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof QueryPage)) return false;

        QueryPage<?> page = (QueryPage<?>) obj;
        return count == page.count && Objects.equals(results, page.results) && resultsField.equals(page.resultsField);
    }

    @Override
    public String toString() {
        return "QueryPage{" +
            "results=" + results +
            ", count=" + count +
            ", resultsField=" + resultsField +
            '}';
    }
}