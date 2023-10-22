package org.elasticsearch.xpack.ml.aggs.kstest;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.plugins.SearchPlugin;
import org.elasticsearch.search.aggregations.AggregationBuilder;
import org.elasticsearch.search.aggregations.BasePipelineAggregationTestCase;
import org.elasticsearch.search.aggregations.bucket.global.GlobalAggregationBuilder;
import org.elasticsearch.search.aggregations.bucket.terms.TermsAggregationBuilder;
import org.elasticsearch.search.aggregations.support.ValueType;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.xpack.ml.MachineLearningTests;

import static org.hamcrest.Matchers.containsString;

public class BucketCountKSTestAggregationBuilderTests extends BasePipelineAggregationTestCase<BucketCountKSTestAggregationBuilder> {

    private static final String AGGREGATION_NAME = "ks-test-agg";

    @Override
    protected List<SearchPlugin> plugins() {
        return List.of(MachineLearningTests.createTrialLicensedMachineLearning(Settings.EMPTY));
    }

    @Override
    protected BucketCountKSTestAggregationBuilder createTestAggregatorFactory() {
        List<Double> values = Stream.generate(ESTestCase::randomDouble).limit(100).collect(Collectors.toList());
        List<String> alternatives = Stream.generate(() -> randomFrom(Alternative.GREATER, Alternative.LESS, Alternative.TWO_SIDED))
            .limit(4)
            .map(Alternative::toString)
            .collect(Collectors.toList());
        SamplingMethod samplingMethod = randomFrom(new SamplingMethod.UpperTail(), new SamplingMethod.LowerTail(), new SamplingMethod.Uniform());
        return new BucketCountKSTestAggregationBuilder(
            AGGREGATION_NAME,
            randomAlphaOfLength(10),
            values,
            alternatives,
            samplingMethod
        );
    }

    public void testValidate() {
        Set<AggregationBuilder> aggBuilders = createAggregationBuilders();
        BucketCountKSTestAggregationBuilder aggBuilderSingleBucket = createAggBuilderSingleBucket(aggBuilders);
        BucketCountKSTestAggregationBuilder aggBuilderMissingAgg = createAggBuilderMissingAgg(aggBuilders);

        // First try to point to a non-existent agg
        assertThat(
            validate(aggBuilders, aggBuilderMissingAgg),
            containsString("aggregation does not exist for aggregation")
        );

        // Now validate with a single bucket agg
        assertThat(
            validate(aggBuilders, aggBuilderSingleBucket),
            containsString("Unable to find unqualified multi-bucket aggregation in buckets_path")
        );
    }

    private Set<AggregationBuilder> createAggregationBuilders() {
        Set<AggregationBuilder> aggBuilders = new HashSet<>();
        aggBuilders.add(new GlobalAggregationBuilder("global"));
        aggBuilders.add(new TermsAggregationBuilder("terms").userValueTypeHint(ValueType.STRING));
        return aggBuilders;
    }

    private BucketCountKSTestAggregationBuilder createAggBuilderSingleBucket(Set<AggregationBuilder> aggBuilders) {
        String qualifiedMetricName = "global>metric";
        List<String> alternatives = Stream.generate(() -> randomFrom(
            Alternative.GREATER,
            Alternative.LESS,
            Alternative.TWO_SIDED)
        ).limit(4).map(Alternative::toString).collect(Collectors.toList());
        List<Double> values = Stream.generate(ESTestCase::randomDouble).limit(100).collect(Collectors.toList());
        SamplingMethod samplingMethod = new SamplingMethod.UpperTail();
        return new BucketCountKSTestAggregationBuilder(AGGREGATION_NAME, qualifiedMetricName, values, alternatives, samplingMethod);
    }

    private BucketCountKSTestAggregationBuilder createAggBuilderMissingAgg(Set<AggregationBuilder> aggBuilders) {
        String qualifiedMetricName = "missing>metric";
        List<String> alternatives = Stream.generate(() -> randomFrom(
            Alternative.GREATER,
            Alternative.LESS,
            Alternative.TWO_SIDED)
        ).limit(4).map(Alternative::toString).collect(Collectors.toList());
        List<Double> values = Stream.generate(ESTestCase::randomDouble).limit(100).collect(Collectors.toList());
        SamplingMethod samplingMethod = new SamplingMethod.UpperTail();
        return new BucketCountKSTestAggregationBuilder(AGGREGATION_NAME, qualifiedMetricName, values, alternatives, samplingMethod);
    }

}