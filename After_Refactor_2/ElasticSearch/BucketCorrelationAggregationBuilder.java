package org.elasticsearch.xpack.ml.aggs.correlation;

import java.io.IOException;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

import org.elasticsearch.TransportVersion;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.search.aggregations.pipeline.BucketHelpers;
import org.elasticsearch.search.aggregations.pipeline.BucketMetricsPipelineAggregationBuilder;
import org.elasticsearch.search.aggregations.pipeline.PipelineAggregator;
import org.elasticsearch.xcontent.ConstructingObjectParser;
import org.elasticsearch.xcontent.ObjectParser;
import org.elasticsearch.xcontent.ParseField;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentParser;
import org.elasticsearch.xpack.core.ml.utils.NamedXContentObjectHelper;

public class BucketCorrelationAggregationBuilder extends BucketMetricsPipelineAggregationBuilder<BucketCorrelationAggregationBuilder> {

    private static final String AGGREGATION_NAME = "bucket_correlation";
    private static final String FUNCTION_FIELD = "function";
    
    private static final String GAP_POLICY_FIELD = "gap_policy";
    private static final String GAP_POLICY_INSERT_ZEROS = BucketHelpers.GapPolicy.INSERT_ZEROS.getName();
    
    private static final ConstructingObjectParser<BucketCorrelationAggregationBuilder, String> PARSER = new ConstructingObjectParser<>(
        AGGREGATION_NAME,
        false,
        (args, context) -> new BucketCorrelationAggregationBuilder(
            context,
            args[0].toString(),
            (CorrelationFunction) args[1],
            (BucketHelpers.GapPolicy) args[2]
        )
    );
    
    private final CorrelationFunction correlationFunction;

    public static final ParseField BUCKETS_PATH_FIELD = new ParseField("buckets_path");
    public static final ParseField NAME_FIELD = new ParseField(AGGREGATION_NAME);

    static {
        PARSER.declareString(ConstructingObjectParser.constructorArg(), BUCKETS_PATH_FIELD);
        PARSER.declareNamedObject(ConstructingObjectParser.constructorArg(), (p, c, n) -> p.namedObject(CorrelationFunction.class, n, null), FUNCTION_FIELD);
        PARSER.declareField(ConstructingObjectParser.optionalConstructorArg(), p -> BucketHelpers.GapPolicy.parse(p.text().toLowerCase(Locale.ROOT), p.getTokenLocation()), GAP_POLICY_FIELD, ObjectParser.ValueType.STRING);
    }

    public BucketCorrelationAggregationBuilder(String name, String bucketsPath, CorrelationFunction correlationFunction) {
        this(name, bucketsPath, correlationFunction, BucketHelpers.GapPolicy.INSERT_ZEROS);
    }

    private BucketCorrelationAggregationBuilder(String name, String bucketsPath, CorrelationFunction correlationFunction, BucketHelpers.GapPolicy gapPolicy) {
        super(name, NAME_FIELD.getPreferredName(), new String[] { bucketsPath }, null, gapPolicy == null ? BucketHelpers.GapPolicy.INSERT_ZEROS : gapPolicy);

        if (gapPolicy != null && !gapPolicy.equals(BucketHelpers.GapPolicy.INSERT_ZEROS)) {
            throw new IllegalArgumentException("only [" + GAP_POLICY_INSERT_ZEROS + "] gap policy is supported");
        }

        this.correlationFunction = correlationFunction;
    }

    public BucketCorrelationAggregationBuilder(StreamInput in) throws IOException {
        super(in, NAME_FIELD.getPreferredName());
        this.correlationFunction = in.readNamedWriteable(CorrelationFunction.class);
    }

    @Override
    public String getWriteableName() {
        return NAME_FIELD.getPreferredName();
    }

    @Override
    protected void innerWriteTo(StreamOutput out) throws IOException {
        out.writeNamedWriteable(correlationFunction);
    }

    @Override
    protected PipelineAggregator createInternal(Map<String, Object> metadata) {
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
    public boolean equals(Object o) {
        if (!super.equals(o)) return false;
        BucketCorrelationAggregationBuilder that = (BucketCorrelationAggregationBuilder) o;
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
}