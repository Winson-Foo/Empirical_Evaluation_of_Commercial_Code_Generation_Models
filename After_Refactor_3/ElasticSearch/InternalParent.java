// This class represents the results of the ChildrenToParentAggregator
public class ParentAggregation extends InternalSingleBucketAggregation implements Parent {

    public ParentAggregation(String name, long count, InternalAggregations subAggs, Map<String, Object> metadata) {
        super(name, docCount, subAggs, metadata);
    }

    // Constructor to read from StreamInput
    public ParentAggregation(StreamInput in) throws IOException {
        super(in);
    }

    // Returns the name of the aggregation
    @Override
    public String getWriteableName() {
        return ParentAggregationBuilder.NAME;
    }

    // Creates a new ParentAggregation with the given params
    @Override
    protected InternalSingleBucketAggregation newAggregation(String name, long count, InternalAggregations subAggs) {
        return new ParentAggregation(name, count, subAggs, getMetadata());
    }
}