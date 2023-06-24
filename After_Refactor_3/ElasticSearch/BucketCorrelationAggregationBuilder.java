package org.elasticsearch.xpack.ml.aggs.correlation;

import java.io.IOException;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.search.aggregations.pipeline.BucketHelpers;
import org.elasticsearch.search.aggregations.pipeline.BucketMetricsPipelineAggregationBuilder;
import org.elasticsearch.search.aggregations.pipeline.PipelineAggregator;
import org.elasticsearch.xcontent.ConstructingObjectParser;
import org.elasticsearch.xcontent.NamedXContentRegistry;
import org.elasticsearch.xcontent.ObjectParser;
import org.elasticsearch.xcontent.ParseField;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentParser;
import org.elasticsearch.xpack.core.ml.utils.NamedXContentObjectHelper;

public class BucketCorrelationAggregationBuilder
        extends BucketMetricsPipelineAggregationBuilder<BucketCorrelationAggregationBuilder> {

    private static final String TYPE = "bucket_correlation";
    private static final String FUNCTION_FIELD = "function";
    private static final String GAP_POLICY_FIELD = "gap_policy";
    private static final BucketHelpers.GapPolicy DEFAULT_GAP_POLICY = BucketHelpers.GapPolicy.INSERT_ZEROS;

    private final CorrelationFunction correlationFunction;

    private BucketCorrelationAggregationBuilder(String name, String[] bucketsPaths,
            CorrelationFunction correlationFunction, BucketHelpers.GapPolicy gapPolicy) {
        super(name, TYPE, bucketsPaths, null, gapPolicy == null ? DEFAULT_GAP_POLICY : gapPolicy);
        this.correlationFunction = Objects.requireNonNull(correlationFunction);
    }

    public BucketCorrelationAggregationBuilder(String name, String bucketsPath,
            CorrelationFunction correlationFunction) {
        this(name, new String[] { bucketsPath }, correlationFunction, null);
    }

    @SuppressWarnings("unchecked")
    public static ObjectParser<BucketCorrelationAggregationBuilder, NamedXContentRegistry> parser(
            NamedXContentRegistry xContentRegistry) {
        return new ObjectParser<>(TYPE,
                ConstructingObjectParser.<BucketCorrelationAggregationBuilder> builder(
                        (p, c) -> new BucketCorrelationAggregationBuilder(
                                p.get(BUCKET_PATH_FIELD.getPreferredName()),
                                new String[] { p.get(BUCKET_PATH_FIELD.getPreferredName()) },
                                p.getParser().namedObject(CorrelationFunction.class, FUNCTION_FIELD, null),
                                p.get(GAP_POLICY_FIELD) != null
                                        ? BucketHelpers.GapPolicy.parse(p.get(GAP_POLICY_FIELD),
                                                p.getTokenLocation())
                                        : null)));
    }

    public BucketCorrelationAggregationBuilder(StreamInput in) throws IOException {
        super(in, TYPE);
        this.correlationFunction = in.readNamedWriteable(CorrelationFunction.class);
    }

    @Override
    protected void innerWriteTo(StreamOutput out) throws IOException {
        out.writeNamedWriteable(correlationFunction);
    }

    @Override
    protected PipelineAggregator doBuild(Map<String, Object> metadata) {
        return new BucketCorrelationAggregator(name, correlationFunction, bucketsPaths[0], metadata);
    }

    @Override
    protected boolean overrideBucketsPath() {
        return true;
    }

    @Override
    protected XContentBuilder doXContentBody(XContentBuilder builder, Params params) throws IOException {
        builder.field(BUCKETS_PATH_FIELD.getPreferredName(), bucketsPaths[0]);
        NamedXContentObjectHelper.writeNamedObject(builder, params, FUNCTION_FIELD, correlationFunction);
        return builder;
    }

    @Override
    protected void validate(ValidationContext context) {
        super.validate(context);
        correlationFunction.validate(context, bucketsPaths[0]);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (!super.equals(obj))
            return false;
        if (getClass() != obj.getClass())
            return false;
        BucketCorrelationAggregationBuilder other = (BucketCorrelationAggregationBuilder) obj;
        return Objects.equals(correlationFunction, other.correlationFunction);
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), correlationFunction);
    }

    public static class Fields {
        public static final ParseField FUNCTION = new ParseField(FUNCTION_FIELD);
        public static final ParseField GAP_POLICY = new ParseField(GAP_POLICY_FIELD);
    }
}