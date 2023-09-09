package org.elasticsearch.search.aggregations.support;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class ValueTypeTests {

    private static final String STRING = "string";
    private static final String FLOAT = "float";
    private static final String DOUBLE = "double";
    private static final String BYTE = "byte";
    private static final String SHORT = "short";
    private static final String INTEGER = "integer";
    private static final String LONG = "long";
    private static final String DATE = "date";
    private static final String IP = "ip";
    private static final String BOOLEAN = "boolean";

    @Test
    public void testResolve() {
        assertTrue(ValueType.lenientParse(STRING) == ValueType.STRING);
        assertTrue(ValueType.lenientParse(FLOAT) == ValueType.DOUBLE);
        assertTrue(ValueType.lenientParse(DOUBLE) == ValueType.DOUBLE);
        assertTrue(ValueType.lenientParse(BYTE) == ValueType.LONG);
        assertTrue(ValueType.lenientParse(SHORT) == ValueType.LONG);
        assertTrue(ValueType.lenientParse(INTEGER) == ValueType.LONG);
        assertTrue(ValueType.lenientParse(LONG) == ValueType.LONG);
        assertTrue(ValueType.lenientParse(DATE) == ValueType.DATE);
        assertTrue(ValueType.lenientParse(IP) == ValueType.IP);
        assertTrue(ValueType.lenientParse(BOOLEAN) == ValueType.BOOLEAN);
    }

    @Test
    public void testCompatibility() {
        assertTrue(ValueType.DOUBLE.isA(ValueType.NUMERIC));
        assertTrue(ValueType.DOUBLE.isA(ValueType.NUMBER));
        assertTrue(ValueType.DOUBLE.isA(ValueType.LONG));
        assertTrue(ValueType.DOUBLE.isA(ValueType.BOOLEAN));
        assertTrue(ValueType.DOUBLE.isA(ValueType.DATE));
        assertTrue(ValueType.DOUBLE.isA(ValueType.DOUBLE));

        assertTrue(ValueType.LONG.isA(ValueType.NUMERIC));
        assertTrue(ValueType.LONG.isA(ValueType.NUMBER));
        assertTrue(ValueType.LONG.isA(ValueType.LONG));
        assertTrue(ValueType.LONG.isA(ValueType.BOOLEAN));
        assertTrue(ValueType.LONG.isA(ValueType.DATE));
        assertTrue(ValueType.LONG.isA(ValueType.DOUBLE));

        assertTrue(ValueType.DATE.isA(ValueType.NUMERIC));
        assertTrue(ValueType.DATE.isA(ValueType.NUMBER));
        assertTrue(ValueType.DATE.isA(ValueType.LONG));
        assertTrue(ValueType.DATE.isA(ValueType.BOOLEAN));
        assertTrue(ValueType.DATE.isA(ValueType.DATE));
        assertTrue(ValueType.DATE.isA(ValueType.DOUBLE));

        assertTrue(ValueType.NUMERIC.isA(ValueType.NUMERIC));
        assertTrue(ValueType.NUMERIC.isA(ValueType.NUMBER));
        assertTrue(ValueType.NUMERIC.isA(ValueType.LONG));
        assertTrue(ValueType.NUMERIC.isA(ValueType.BOOLEAN));
        assertTrue(ValueType.NUMERIC.isA(ValueType.DATE));
        assertTrue(ValueType.NUMERIC.isA(ValueType.DOUBLE));

        assertTrue(ValueType.BOOLEAN.isA(ValueType.NUMERIC));
        assertTrue(ValueType.BOOLEAN.isA(ValueType.NUMBER));
        assertTrue(ValueType.BOOLEAN.isA(ValueType.LONG));
        assertTrue(ValueType.BOOLEAN.isA(ValueType.BOOLEAN));
        assertTrue(ValueType.BOOLEAN.isA(ValueType.DATE));
        assertTrue(ValueType.BOOLEAN.isA(ValueType.DOUBLE));

        assertFalse(ValueType.STRING.isA(ValueType.NUMBER));
        assertFalse(ValueType.DATE.isA(ValueType.IP));

        assertTrue(ValueType.IP.isA(ValueType.STRING));
        assertTrue(ValueType.STRING.isA(ValueType.IP));
    }
}