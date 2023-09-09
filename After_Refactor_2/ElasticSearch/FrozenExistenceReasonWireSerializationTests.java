import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.test.AbstractWireSerializingTestCase;
import java.util.ArrayList;
import java.util.List;

public class FrozenExistenceReasonSerializerTests extends AbstractWireSerializingTestCase<FrozenExistenceDeciderService.FrozenExistenceReason> {

    @Override
    protected Writeable.Reader<FrozenExistenceDeciderService.FrozenExistenceReason> instanceReader() {
        return FrozenExistenceDeciderService.FrozenExistenceReason::new;
    }

    @Override
    protected FrozenExistenceDeciderService.FrozenExistenceReason createTestInstance() {
        List<String> indices = new ArrayList<>();
        int numIndices = randomIntBetween(0, 10);
        for (int i = 0; i < numIndices; i++) {
            indices.add(randomAlphaNumeric(5));
        }
        return new FrozenExistenceDeciderService.FrozenExistenceReason(indices);
    }

    @Override
    protected FrozenExistenceDeciderService.FrozenExistenceReason mutateInstance(FrozenExistenceDeciderService.FrozenExistenceReason instance) {
        if (randomBoolean()) {
            instance.addIndex(randomAlphaNumeric(5));
        } else if (!instance.isEmpty()) {
            instance.removeIndex(randomIntBetween(0, instance.size() - 1));
        }
        return new FrozenExistenceDeciderService.FrozenExistenceReason(instance.indices());
    }
} 