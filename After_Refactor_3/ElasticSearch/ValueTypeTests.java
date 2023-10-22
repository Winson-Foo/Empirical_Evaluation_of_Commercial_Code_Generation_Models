package org.elasticsearch.search.aggregations.support;

import org.elasticsearch.test.ESTestCase;

public class ValueTypeTests extends ESTestCase {

    public void testValueTypeParsing() {
        // Test string parsing
        assertEquals(ValueType.STRING, ValueType.lenientParse(ValueType.STRING.toString()));

        // Test double parsing
        assertEquals(ValueType.DOUBLE, ValueType.lenientParse(ValueType.FLOAT.toString()));
        assertEquals(ValueType.DOUBLE, ValueType.lenientParse(ValueType.DOUBLE.toString()));

        // Test long parsing
        assertEquals(ValueType.LONG, ValueType.lenientParse(ValueType.BYTE.toString()));
        assertEquals(ValueType.LONG, ValueType.lenientParse(ValueType.SHORT.toString()));
        assertEquals(ValueType.LONG, ValueType.lenientParse(ValueType.INTEGER.toString()));
        assertEquals(ValueType.LONG, ValueType.lenientParse(ValueType.LONG.toString()));

        // Test date parsing
        assertEquals(ValueType.DATE, ValueType.lenientParse(ValueType.DATE.toString()));

        // Test IP parsing
        assertEquals(ValueType.IP, ValueType.lenientParse(ValueType.IP.toString()));

        // Test boolean parsing
        assertEquals(ValueType.BOOLEAN, ValueType.lenientParse(ValueType.BOOLEAN.toString()));
    }

    public void testValueTypeCompatibility() {
        // Test double compatibility
        assertTrue(ValueType.DOUBLE.isA(ValueType.NUMERIC));
        assertTrue(ValueType.DOUBLE.isA(ValueType.NUMBER));
        assertTrue(ValueType.DOUBLE.isA(ValueType.LONG));
        assertTrue(ValueType.DOUBLE.isA(ValueType.BOOLEAN));
        assertTrue(ValueType.DOUBLE.isA(ValueType.DATE));
        assertTrue(ValueType.DOUBLE.isA(ValueType.DOUBLE));

        // Test long compatibility
        assertTrue(ValueType.LONG.isA(ValueType.NUMERIC));
        assertTrue(ValueType.LONG.isA(ValueType.NUMBER));
        assertTrue(ValueType.LONG.isA(ValueType.LONG));
        assertTrue(ValueType.LONG.isA(ValueType.BOOLEAN));
        assertTrue(ValueType.LONG.isA(ValueType.DATE));
        assertTrue(ValueType.LONG.isA(ValueType.DOUBLE));

        // Test date compatibility
        assertTrue(ValueType.DATE.isA(ValueType.NUMERIC));
        assertTrue(ValueType.DATE.isA(ValueType.NUMBER));
        assertTrue(ValueType.DATE.isA(ValueType.LONG));
        assertTrue(ValueType.DATE.isA(ValueType.BOOLEAN));
        assertTrue(ValueType.DATE.isA(ValueType.DATE));
        assertTrue(ValueType.DATE.isA(ValueType.DOUBLE));

        // Test numeric compatibility
        assertTrue(ValueType.NUMERIC.isA(ValueType.NUMERIC));
        assertTrue(ValueType.NUMERIC.isA(ValueType.NUMBER));
        assertTrue(ValueType.NUMERIC.isA(ValueType.LONG));
        assertTrue(ValueType.NUMERIC.isA(ValueType.BOOLEAN));
        assertTrue(ValueType.NUMERIC.isA(ValueType.DATE));
        assertTrue(ValueType.NUMERIC.isA(ValueType.DOUBLE));

        // Test boolean compatibility
        assertTrue(ValueType.BOOLEAN.isA(ValueType.NUMERIC));
        assertTrue(ValueType.BOOLEAN.isA(ValueType.NUMBER));
        assertTrue(ValueType.BOOLEAN.isA(ValueType.LONG));
        assertTrue(ValueType.BOOLEAN.isA(ValueType.BOOLEAN));
        assertTrue(ValueType.BOOLEAN.isA(ValueType.DATE));
        assertTrue(ValueType.BOOLEAN.isA(ValueType.DOUBLE));

        // Test string and IP compatibility
        assertFalse(ValueType.STRING.isA(ValueType.NUMBER));
        assertFalse(ValueType.DATE.isA(ValueType.IP));

        assertTrue(ValueType.IP.isA(ValueType.STRING));
        assertTrue(ValueType.STRING.isA(ValueType.IP));
    }
}