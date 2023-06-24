package org.elasticsearch.search.aggregations.pipeline;

import java.util.function.Function;
import java.util.function.IntToDoubleFunction;

import static org.elasticsearch.search.aggregations.PipelineAggregatorBuilders.statsBucket;
import static org.hamcrest.Matchers.equalTo;

public class StatsBucketIT extends BucketMetricsPipeLineAggregationTestCase<StatsBucket> {

    // Constants
    private static final int NUM_BUCKETS = 10;
    private static final double POSITIVE_INFINITY = Double.POSITIVE_INFINITY;
    private static final double NEGATIVE_INFINITY = Double.NEGATIVE_INFINITY;
    private static final double NAN = Double.NaN;

    // Instance variables
    private IntToDoubleFunction bucketValues;
    private Function<Integer, String> bucketKeys;
    private StatsBucket pipelineBucket;

    // Test cases
    @Override
    protected StatsBucketPipelineAggregationBuilder BucketMetricsPipelineAgg(String name, String bucketsPath) {
        return statsBucket(name, bucketsPath);
    }

    @Override
    protected void assertResult(IntToDoubleFunction bucketValues, Function<Integer, String> bucketKeys,
            int numBuckets, StatsBucket pipelineBucket) {
        // Input validation
        if (bucketValues == null || bucketKeys == null || pipelineBucket == null) {
            throw new IllegalArgumentException("Arguments cannot be null");
        }

        this.bucketValues = bucketValues;
        this.bucketKeys = bucketKeys;
        this.pipelineBucket = pipelineBucket;

        // Compute expected results
        double sum = 0;
        int count = 0;
        double min = POSITIVE_INFINITY;
        double max = NEGATIVE_INFINITY;
        for (int i = 0; i < numBuckets; ++i) {
            double bucketValue = bucketValues.applyAsDouble(i);
            count++;
            sum += bucketValue;
            min = Math.min(min, bucketValue);
            max = Math.max(max, bucketValue);
        }

        double avgValue = count == 0 ? NAN : (sum / count);

        // Assertions
        assertThat(pipelineBucket.getAvg(), equalTo(avgValue));
        assertThat(pipelineBucket.getMin(), equalTo(min));
        assertThat(pipelineBucket.getMax(), equalTo(max));
    }

    // Helper methods
    @Override
    protected String nestedMetric() {
        return "avg";
    }

    @Override
    protected double getNestedMetric(StatsBucket bucket) {
        return bucket.getAvg();
    }
}