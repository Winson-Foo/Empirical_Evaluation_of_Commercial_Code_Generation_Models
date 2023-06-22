/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ml.aggs.kstest;

import java.util.List;
import java.util.Set;
import java.util.HashSet;
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

    private static final String NAME = "ks-test-agg";
    private static final int NUM_SAMPLES = 100;
    private static final int NUM_ALTERNATIVES = 4;
    private static final String GLOBAL_AGG_NAME = "global";
    private static final String TERM_AGG_NAME = "terms";
    private static final String METRIC_NAME_PREFIX = "metric";
    private static final SamplingMethod[] SAMPLING_METHODS = new SamplingMethod[] { 
        new SamplingMethod.UpperTail(), 
        new SamplingMethod.LowerTail(), 
        new SamplingMethod.Uniform() 
    };

    @Override
    protected List<SearchPlugin> plugins() {
        return List.of(MachineLearningTests.createTrialLicensedMachineLearning(Settings.EMPTY));
    }

    @Override
    protected BucketCountKSTestAggregationBuilder createTestAggregatorFactory() {
        List<Double> samples = Stream.generate(ESTestCase::randomDouble).limit(NUM_SAMPLES).collect(Collectors.toList());
        List<String> alternativeNames = Stream.generate(() -> randomFrom(Alternative.GREATER, Alternative.LESS, Alternative.TWO_SIDED))
                .limit(NUM_ALTERNATIVES)
                .map(Alternative::toString)
                .collect(Collectors.toList());
        SamplingMethod samplingMethod = randomFrom(SAMPLING_METHODS);

        return new BucketCountKSTestAggregationBuilder(
            NAME,
            randomAlphaOfLength(10),
            samples,
            alternativeNames,
            samplingMethod
        );
    }
    
    public void testValidate() {
        Set<AggregationBuilder> aggBuilders = new HashSet<>();
        AggregationBuilder singleBucketAgg = new GlobalAggregationBuilder(GLOBAL_AGG_NAME);
        AggregationBuilder multiBucketAgg = new TermsAggregationBuilder(TERM_AGG_NAME).userValueTypeHint(ValueType.STRING);

        aggBuilders.add(singleBucketAgg);
        aggBuilders.add(multiBucketAgg);

        // Validate that non-existent aggregation returns correct error message
        assertThat(
            validate(
                aggBuilders,
                new BucketCountKSTestAggregationBuilder(
                    NAME,
                    "missing>" + METRIC_NAME_PREFIX,
                    Stream.generate(ESTestCase::randomDouble).limit(NUM_SAMPLES).collect(Collectors.toList()),
                    Stream.generate(() -> randomFrom(Alternative.GREATER, Alternative.LESS, Alternative.TWO_SIDED))
                        .limit(NUM_ALTERNATIVES)
                        .map(Alternative::toString)
                        .collect(Collectors.toList()),
                    randomFrom(SAMPLING_METHODS)
                )
            ),
            containsString("aggregation does not exist for aggregation")
        );

        // Validate that single bucket aggregation returns correct error message
        assertThat(
            validate(
                aggBuilders,
                new BucketCountKSTestAggregationBuilder(
                    NAME,
                    GLOBAL_AGG_NAME + ">" + METRIC_NAME_PREFIX,
                    Stream.generate(ESTestCase::randomDouble).limit(NUM_SAMPLES).collect(Collectors.toList()),
                    Stream.generate(() -> randomFrom(Alternative.GREATER, Alternative.LESS, Alternative.TWO_SIDED))
                        .limit(NUM_ALTERNATIVES)
                        .map(Alternative::toString)
                        .collect(Collectors.toList()),
                    randomFrom(SAMPLING_METHODS)
                )
            ),
            containsString("Unable to find unqualified multi-bucket aggregation in buckets_path")
        );
    }

}