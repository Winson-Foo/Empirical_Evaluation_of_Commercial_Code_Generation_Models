package org.elasticsearch.xpack.core.action.util;

import java.util.List;
import java.util.Objects;

import org.elasticsearch.ResourceNotFoundException;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.xcontent.ToXContent;
import org.elasticsearch.xcontent.ToXContentObject;
import org.elasticsearch.xcontent.XContentBuilder;

import java.io.IOException;

public class QueryPage<T> implements ToXContentObject, Writeable {
    private final String resultsField;
    private final List<T> results;
    private final long count;

    public QueryPage(List<T> results, long count, String resultsField) {
        this.results = Objects.requireNonNull(results);
        this.count = count;
        this.resultsField = Objects.requireNonNull(resultsField);
    }

    public QueryPage(StreamInput in, Reader<T> reader) throws IOException {
        this(in.readList(reader), in.readLong(), in.readString());
    }

    public static ResourceNotFoundException emptyQueryPage(String resultsField) {
        return new ResourceNotFoundException("Could not find requested " + resultsField);
    }

    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(resultsField);
        out.writeList(results);
        out.writeLong(count);
    }

    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field("count", count);
        builder.startArray(resultsField);
        for (T result : results) {
            if (result != null) {
                ((ToXContent) result).toXContent(builder, params);
            }
        }
        builder.endArray();
        builder.endObject();
        return builder;
    }

    public List<T> getResults() {
        return results;
    }

    public long getCount() {
        return count;
    }

    public String getResultsField() {
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