/**
 * Represents the result of a parent aggregation.
 */
public class ParentAggregationResult extends InternalSingleBucketAggregation implements Parent {
  
    private final Map<String, Object> metadata;

    public ParentAggregationResult(String name, long docCount, InternalAggregations aggregations, Map<String, Object> metadata) {
        super(name, docCount, aggregations, metadata);
        this.metadata = metadata;
    }

    /**
     * Read from a stream.
     */
    public ParentAggregationResult(StreamInput in) throws IOException {
        super(in);
        this.metadata = in.readMap();
    }

    /**
     * Gets the name of the aggregation.
     */
    @Override
    public String getWriteableName() {
        return ParentAggregationBuilder.NAME;
    }

    /**
     * Creates a new instance of the aggregation.
     */
    @Override
    protected InternalSingleBucketAggregation newAggregation(String name, long docCount, InternalAggregations subAggregations) {
        return new ParentAggregationResult(name, docCount, subAggregations, metadata);
    }

    /**
     * Gets the metadata associated with the aggregation.
     */
    public Map<String, Object> getMetadata() {
        return metadata;
    }
}