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
import static java.util.stream.Collectors.toList;

public class BucketCountKSTestAggregationBuilderTests extends BasePipelineAggregationTestCase<BucketCountKSTestAggregationBuilder> {

    private static final String NAME = "ks-test-agg";
    private static final String METRIC = "metric";
    private static final List<Alternative> ALTERNATIVES = Stream.generate(() -> ESTestCase.randomFrom(Alternative.values()))
        .limit(4)
        .collect(toList());

    @Override
    protected List<SearchPlugin> plugins() {
        return List.of(MachineLearningTests.createTrialLicensedMachineLearning(Settings.EMPTY));
    }

    @Override
    protected BucketCountKSTestAggregationBuilder createTestAggregatorFactory() {
        List<Double> randomDoubles = Stream.generate(ESTestCase::randomDouble).limit(100).collect(toList());
        SamplingMethod samplingMethod = ESTestCase.randomFrom(new SamplingMethod.UpperTail(), new SamplingMethod.LowerTail(), new SamplingMethod.Uniform());

        return new BucketCountKSTestAggregationBuilder(
            NAME,
            randomAlphaOfLength(10),
            randomDoubles,
            ALTERNATIVES.stream().map(Enum::toString).collect(toList()),
            samplingMethod
        );
    }

    public void testValidateWithNonexistentAgg() {
        String missingMetric = "missing>metric";
        BucketCountKSTestAggregationBuilder builder = new BucketCountKSTestAggregationBuilder(
            NAME,
            missingMetric,
            Stream.generate(ESTestCase::randomDouble).limit(100).collect(toList()),
            ALTERNATIVES.stream().map(Enum::toString).collect(toList()),
            new SamplingMethod.UpperTail()
        );
        Set<AggregationBuilder> aggBuilders = new HashSet<>();
        GlobalAggregationBuilder singleBucketAgg = new GlobalAggregationBuilder("global");
        TermsAggregationBuilder multiBucketAgg = new TermsAggregationBuilder("terms").userValueTypeHint(ValueType.STRING);
        aggBuilders.add(singleBucketAgg);
        aggBuilders.add(multiBucketAgg);

        String errorMessage = validate(aggBuilders, builder);

        assertThat(errorMessage, containsString("aggregation does not exist for aggregation"));
        assertThat(errorMessage, containsString(missingMetric));
    }

    public void testValidateWithSingleBucketAgg() {
        String globalMetric = "global>metric";
        BucketCountKSTestAggregationBuilder builder = new BucketCountKSTestAggregationBuilder(
            NAME,
            globalMetric,
            Stream.generate(ESTestCase::randomDouble).limit(100).collect(toList()),
            ALTERNATIVES.stream().map(Enum::toString).collect(toList()),
            new SamplingMethod.UpperTail()
        );
        Set<AggregationBuilder> aggBuilders = new HashSet<>();
        GlobalAggregationBuilder singleBucketAgg = new GlobalAggregationBuilder("global");
        TermsAggregationBuilder multiBucketAgg = new TermsAggregationBuilder("terms").userValueTypeHint(ValueType.STRING);
        aggBuilders.add(singleBucketAgg);
        aggBuilders.add(multiBucketAgg);

        String errorMessage = validate(aggBuilders, builder);

        assertThat(errorMessage, containsString("Unable to find unqualified multi-bucket aggregation in buckets_path"));
        assertThat(errorMessage, containsString(globalMetric));
    }

}