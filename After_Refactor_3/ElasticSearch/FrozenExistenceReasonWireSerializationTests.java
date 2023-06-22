/**
 * Tests for the FrozenExistenceDeciderService.
 */
public class FrozenExistenceDeciderServiceTest extends AbstractWireSerializingTestCase<FrozenExistenceReason> {

    @Override
    protected Writeable.Reader<FrozenExistenceReason> instanceReader() {
        return FrozenExistenceReason::new;
    }

    @Override
    protected FrozenExistenceReason createTestInstance() {
        return new FrozenExistenceReason(randomList(between(0, 10), () -> randomAlphaOfLength(5)));
    }

    @Override
    protected FrozenExistenceReason mutateInstance(FrozenExistenceReason instance) {
        List<String> indices = new ArrayList<>(instance.getIndices());
        if (indices.isEmpty() || randomBoolean()) {
            indices.add(randomAlphaOfLength(5));
        } else {
            indices.remove(between(0, indices.size() - 1));
        }
        return new FrozenExistenceReason(indices);
    }
    
    @Test
    public void testEqualsAndHashCode() {
        // Create two instances with the same indices
        List<String> indices = List.of("index1", "index2");
        FrozenExistenceReason reason1 = new FrozenExistenceReason(indices);
        FrozenExistenceReason reason2 = new FrozenExistenceReason(indices);
        
        // Test that they are equal and have the same hash code
        assertEquals(reason1, reason2);
        assertEquals(reason1.hashCode(), reason2.hashCode());
    }
    
    @ParameterizedTest
    @MethodSource("provideReasonAndExpectedString")
    public void testToString(FrozenExistenceReason reason, String expectedString) {
        assertEquals(expectedString, reason.toString());
    }
    
    private static Stream<Arguments> provideReasonAndExpectedString() {
        List<String> indices1 = List.of("index1", "index2");
        String expectedString1 = "FrozenExistenceReason[indices=[" + String.join(",", indices1) + "]]";
        FrozenExistenceReason reason1 = new FrozenExistenceReason(indices1);
        
        List<String> indices2 = List.of("index3", "index4");
        String expectedString2 = "FrozenExistenceReason[indices=[" + String.join(",", indices2) + "]]";
        FrozenExistenceReason reason2 = new FrozenExistenceReason(indices2);
        
        return Stream.of(
            Arguments.of(reason1, expectedString1),
            Arguments.of(reason2, expectedString2)
        );
    }
}