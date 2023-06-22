public class InternalParent extends InternalSingleBucketAggregation implements Parent {
    public InternalParent(String name, long docCount, InternalAggregations aggregations, Map<String, Object> metadata) {
        super(name, docCount, aggregations, metadata);
    }

    public InternalParent(StreamInput in) throws IOException {
        super(in);
    }

    @Override
    public String getWriteableName() {
        return ParentAggregationBuilder.NAME;
    }

    @Override
    protected InternalSingleBucketAggregation newAggregation(String name, long docCount, InternalAggregations subAggregations) {
        return new InternalParent(name, docCount, subAggregations, getMetadata());
    }
}

// Parent.java

public interface Parent extends InternalSingleBucketAggregation {
}

// ParentAggregationBuilder.java

public class ParentAggregationBuilder extends AbstractAggregationBuilder<ParentAggregationBuilder> {
    public static final String NAME = "parent";

    // Constructor, setters, getters, build() method, etc.
}

// ChildrenToParentAggregator.java

public class ChildrenToParentAggregator extends Aggregator {
    // Fields, constructor, methods, etc.
}

// InternalChildren.java

public class InternalChildren extends InternalSingleBucketAggregation implements Children {
    // Fields, constructor, methods, etc.
}

// ChildrenAggregationBuilder.java

public class ChildrenAggregationBuilder extends AbstractAggregationBuilder<ChildrenAggregationBuilder> {
    public static final String NAME = "children";

    // Constructor, setters, getters, build() method, etc.
}

// ChildrenAggregator.java

public class ChildrenAggregator extends Aggregator {
    // Fields, constructor, methods, etc.
}

// AggregationBuilders.java

public final class AggregationBuilders {
    private AggregationBuilders() {
    }

    public static ChildrenAggregationBuilder children(String name, String type) {
        return new ChildrenAggregationBuilder().name(name).type(type);
    }

    public static ParentAggregationBuilder parent(String name, String type) {
        return new ParentAggregationBuilder().name(name).type(type);
    }

    // Other static factory methods for creating aggregation builders
}