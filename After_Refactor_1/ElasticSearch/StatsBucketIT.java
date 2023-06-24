package org.elasticsearch.search.aggregations.pipeline;

import java.util.function.Function;
import java.util.function.IntToDoubleFunction;

import static org.elasticsearch.search.aggregations.PipelineAggregatorBuilders.statsBucket;
import static org.junit.Assert.assertEquals;

public class StatsBucketTest extends BucketMetricsPipeLineAggregationTestCase<StatsBucket> {

    @Override
    protected StatsBucketPipelineAggregationBuilder createAggregationBuilder(String name, String bucketsPath) {
        return statsBucket(name, bucketsPath);
    }

    @Override
    protected void assertResult(
        IntToDoubleFunction bucketValues,
        Function<Integer, String> bucketKeys,
        int numBuckets,
        StatsBucket pipelineBucket
    ) {
        Stats stats = calculateStats(bucketValues, numBuckets);
        assertEquals(stats.avg, pipelineBucket.getAvg(), 0.00001);
        assertEquals(stats.min, pipelineBucket.getMin(), 0.00001);
        assertEquals(stats.max, pipelineBucket.getMax(), 0.00001);
    }

    private Stats calculateStats(IntToDoubleFunction bucketValues, int numBuckets) {
        double sum = 0;
        int count = 0;
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < numBuckets; ++i) {
            double bucketValue = bucketValues.applyAsDouble(i);
            count++;
            sum += bucketValue;
            min = Math.min(min, bucketValue);
            max = Math.max(max, bucketValue);
        }

        double avg = count == 0 ? Double.NaN : (sum / count);
        return new Stats(avg, min, max);
    }
    
    private static class Stats {
        final double avg;
        final double min;
        final double max;
        Stats(double avg, double min, double max) {
            this.avg = avg;
            this.min = min;
            this.max = max;
        }
    }
}