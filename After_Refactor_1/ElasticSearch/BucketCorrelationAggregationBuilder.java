package org.elasticsearch.xpack.ml.aggs.correlation;

import org.elasticsearch.TransportVersion;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.search.aggregations.pipeline.BucketHelpers;
import org.elasticsearch.search.aggregations.pipeline.BucketMetricsPipelineAggregationBuilder;
import org.elasticsearch.search.aggregations.pipeline.PipelineAggregator;
import org.elasticsearch.xcontent.ConstructingObjectParser;
import org.elasticsearch.xcontent.ObjectParser;
import org.elasticsearch.xcontent.NamedXContentRegistry;
import org.elasticsearch.xcontent.ParseField;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentParser;
import org.elasticsearch.xpack.core.ml.utils.NamedXContentObjectHelper;

import java.io.IOException;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

public class CorrelationAggregationBuilder extends BucketMetricsPipelineAggregationBuilder<CorrelationAggregationBuilder>
        implements CorrelationAggregationBuilder.XContent, CorrelationAggregationBuilder.Validate,
        CorrelationAggregationBuilder.Writeable, CorrelationAggregationBuilder.Internal {

    private static final String NAME = "correlation";
    private static final String BUCKETS_PATH = "buckets_path";
    private static final String FUNCTION = "function";
    private static final String GAP_POLICY = "gap_policy";
    private static final String MESSAGE_ONLY_INSERT_ZEROS_SUPPORTED = "only [%s] of [%s] is supported";

    private final CorrelationFunction correlationFunction;
    private BucketHelpers.GapPolicy gapPolicy = BucketHelpers.GapPolicy.INSERT_ZEROS;

    public static final ObjectParser<CorrelationAggregationBuilder, NamedXContentRegistry> PARSER =
        new ObjectParser<>(NAME, CorrelationAggregationBuilder::new);

    static {
        PARSER.declareString(CorrelationAggregationBuilder::setBucketsPath, new ParseField(BUCKETS_PATH));
        PARSER.declareObject((parser, context) ->
                CorrelationFunction.parse(parser, context), new ParseField(FUNCTION));
        PARSER.declareString((p) -> {
            if (p.equalsIgnoreCase(BucketHelpers.GapPolicy.INSERT_ZEROS.getName())) {
                return;
            }
            throw new IllegalArgumentException(String.format(MESSAGE_ONLY_INSERT_ZEROS_SUPPORTED, GAP_POLICY,
                    BucketHelpers.GapPolicy.INSERT_ZEROS.getName()));
        }, new ParseField(GAP_POLICY));
    }

    public CorrelationAggregationBuilder(String name, String bucketsPath, CorrelationFunction correlationFunction) {
        super(name, NAME, new String[]{bucketsPath}, null, BucketHelpers.GapPolicy.INSERT_ZEROS);
        this.correlationFunction = correlationFunction;
    }

    public CorrelationAggregationBuilder(StreamInput in) throws IOException {
        super(in, NAME);
        this.correlationFunction = in.readNamedWriteable(CorrelationFunction.class);
    }

    public CorrelationAggregationBuilder setFunction(CorrelationFunction correlationFunction) {
        return new CorrelationAggregationBuilder(name, bucketsPaths[0], correlationFunction).setGapPolicy(gapPolicy);
    }

    public CorrelationAggregationBuilder setGapPolicy(String gapPolicyStr) {
        if (gapPolicyStr.equalsIgnoreCase(BucketHelpers.GapPolicy.INSERT_ZEROS.getName())) {
            return setGapPolicy(BucketHelpers.GapPolicy.INSERT_ZEROS);
        }
        throw new IllegalArgumentException(String.format(MESSAGE_ONLY_INSERT_ZEROS_SUPPORTED, GAP_POLICY,
                BucketHelpers.GapPolicy.INSERT_ZEROS.getName()));
    }

    public CorrelationAggregationBuilder setGapPolicy(BucketHelpers.GapPolicy gapPolicy) {
        this.gapPolicy = gapPolicy;
        return this;
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    @Override
    protected void innerWriteTo(StreamOutput out) throws IOException {
        out.writeNamedWriteable(correlationFunction);
    }

    @Override
    protected PipelineAggregator createInternal(Map<String, Object> metadata) {
        return new CorrelationAggregator(name, correlationFunction, bucketsPaths[0], metadata);
    }

    @Override
    protected boolean overrideBucketsPath() {
        return true;
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.field(BUCKETS_PATH, bucketsPaths[0]);
        NamedXContentObjectHelper.writeNamedObject(builder, params, FUNCTION, correlationFunction);
        return builder;
    }

    @Override
    public void validate(ValidationContext context) {
        super.validate(context);
        correlationFunction.validate(context, bucketsPaths[0]);
    }

    @Override
    public boolean equals(Object o) {
        if (super.equals(o) == false) return false;
        CorrelationAggregationBuilder that = (CorrelationAggregationBuilder) o;
        return Objects.equals(correlationFunction, that.correlationFunction);
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), correlationFunction);
    }

    @Override
    public TransportVersion getMinimalSupportedVersion() {
        return TransportVersion.V_7_14_0;
    }

    interface Writeable {
        String getWriteableName();

        void innerWriteTo(StreamOutput out) throws IOException;
    }

    interface Internal {
        PipelineAggregator createInternal(Map<String, Object> metadata);

        boolean overrideBucketsPath();
    }

    interface XContent {
        XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException;
    }

    interface Validate {
        void validate(ValidationContext context);
    }
}

