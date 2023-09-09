package org.elasticsearch.search.aggregations.pipeline;

import java.util.function.Function;
import java.util.function.IntToDoubleFunction;
import static org.elasticsearch.search.aggregations.PipelineAggregatorBuilders.statsBucket;
import static org.hamcrest.Matchers.equalTo;
import org.junit.Test;

public class StatsBucketPipelineIT extends BucketMetricsPipeLineAggregationTestCase<StatsBucket> {

    @Override
    protected StatsBucketPipelineAggregationBuilder BucketMetricsPipelineAgg(String name, String bucketsPath) {
        return statsBucket(name, bucketsPath);
    }

    @Override
    protected String nestedMetric() {
        return "avg";
    }

    @Override
    protected double getNestedMetric(StatsBucket bucket) {
        return bucket.getAvg();
    }

    @Test
    public void testPipelineAggregation() throws Exception {
        // Initialize variables
        final int numBuckets = 5;
        final double[] bucketValues = { 10, 20, 30, 40, 50 };
        final String[] bucketKeys = { "a", "b", "c", "d", "e" };
        final double expectedAvg = 30.0;
        final double expectedMin = 10.0;
        final double expectedMax = 50.0;

        // Create pipeline aggregation
        StatsBucketPipelineAggregationBuilder pipelineAggregation = BucketMetricsPipelineAgg("stats_bucket", "test");

        // Execute pipeline aggregation
        StatsBucket pipelineBucket = executePipelineAggregation(pipelineAggregation, numBuckets, bucketValues, bucketKeys);

        // Assert pipeline aggregation results
        assertThat(pipelineBucket.getAvg(), equalTo(expectedAvg));
        assertThat(pipelineBucket.getMin(), equalTo(expectedMin));
        assertThat(pipelineBucket.getMax(), equalTo(expectedMax));
    }
}

class PipelineUtils {
    /**
     * Returns a pipeline aggregation builder for StatsBucket.
     *
     * @param name The name of the pipeline aggregation.
     * @param bucketsPath The path to the buckets to be aggregated.
     * @return The pipeline aggregation builder.
     */
    public static StatsBucketPipelineAggregationBuilder statsBucket(String name, String bucketsPath) {
        return statsBucket(name, bucketsPath);
    }

    /**
     * Asserts the results of the pipeline aggregation.
     *
     * @param bucketValues A function that returns the bucket value for the given index.
     * @param bucketKeys A function that returns the bucket key for the given index.
     * @param numBuckets The number of buckets to aggregate.
     * @param pipelineBucket The pipeline aggregation results.
     */
    public static void assertPipelineAggResults(IntToDoubleFunction bucketValues, Function<Integer, String> bucketKeys, int numBuckets, StatsBucket pipelineBucket) {
        double sum = 0.0;
        int count = 0;
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;

        // Calculate summary statistics
        for (int i = 0; i < numBuckets; i++) {
            double bucketValue = bucketValues.applyAsDouble(i);
            count++;
            sum += bucketValue;
            min = Math.min(min, bucketValue);
            max = Math.max(max, bucketValue);
        }

        // Calculate expected values
        double expectedAvg = count == 0 ? Double.NaN : (sum / count);
        double expectedMin = min;
        double expectedMax = max;

        // Assert results
        assertThat(pipelineBucket.getAvg(), equalTo(expectedAvg));
        assertThat(pipelineBucket.getMin(), equalTo(expectedMin));
        assertThat(pipelineBucket.getMax(), equalTo(expectedMax));
    }

    /**
     * Returns the name of the nested metric for StatsBucket.
     *
     * @return The name of the nested metric.
     */
    public static String getNestedMetricName() {
        return "avg";
    }

    /**
     * Returns the value of the nested metric for StatsBucket.
     *
     * @param bucket The StatsBucket.
     * @return The value of the nested metric.
     */
    public static double getNestedMetricValue(StatsBucket bucket) {
        return bucket.getAvg();
    }
}