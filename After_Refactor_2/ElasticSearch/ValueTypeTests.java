package org.elasticsearch.search.aggregations.support;

import static org.elasticsearch.search.aggregations.support.ValueType.BOOLEAN;
import static org.elasticsearch.search.aggregations.support.ValueType.DATE;
import static org.elasticsearch.search.aggregations.support.ValueType.DOUBLE;
import static org.elasticsearch.search.aggregations.support.ValueType.IP;
import static org.elasticsearch.search.aggregations.support.ValueType.LONG;
import static org.elasticsearch.search.aggregations.support.ValueType.NUMERIC;
import static org.elasticsearch.search.aggregations.support.ValueType.NUMBER;
import static org.elasticsearch.search.aggregations.support.ValueType.STRING;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class ValueTypeTest {

    @Test
    public void shouldResolveStringValueTypes() {
        assertEquals(STRING, ValueType.lenientParse("string"));
        assertEquals(DOUBLE, ValueType.lenientParse("float"));
        assertEquals(DOUBLE, ValueType.lenientParse("double"));
        assertEquals(LONG, ValueType.lenientParse("byte"));
        assertEquals(LONG, ValueType.lenientParse("short"));
        assertEquals(LONG, ValueType.lenientParse("integer"));
        assertEquals(LONG, ValueType.lenientParse("long"));
        assertEquals(DATE, ValueType.lenientParse("date"));
        assertEquals(IP, ValueType.lenientParse("ip"));
        assertEquals(BOOLEAN, ValueType.lenientParse("boolean"));
    }

    @Test
    public void shouldCheckCompatibilityForValueTypes() {
        assertTrue(DOUBLE.isA(NUMERIC));
        assertTrue(DOUBLE.isA(NUMBER));
        assertTrue(DOUBLE.isA(LONG));
        assertTrue(DOUBLE.isA(BOOLEAN));
        assertTrue(DOUBLE.isA(DATE));
        assertTrue(DOUBLE.isA(DOUBLE));

        assertTrue(LONG.isA(NUMERIC));
        assertTrue(LONG.isA(NUMBER));
        assertTrue(LONG.isA(LONG));
        assertTrue(LONG.isA(BOOLEAN));
        assertTrue(LONG.isA(DATE));
        assertTrue(LONG.isA(DOUBLE));

        assertTrue(DATE.isA(NUMERIC));
        assertTrue(DATE.isA(NUMBER));
        assertTrue(DATE.isA(LONG));
        assertTrue(DATE.isA(BOOLEAN));
        assertTrue(DATE.isA(DATE));
        assertTrue(DATE.isA(DOUBLE));

        assertTrue(NUMERIC.isA(NUMERIC));
        assertTrue(NUMERIC.isA(NUMBER));
        assertTrue(NUMERIC.isA(LONG));
        assertTrue(NUMERIC.isA(BOOLEAN));
        assertTrue(NUMERIC.isA(DATE));
        assertTrue(NUMERIC.isA(DOUBLE));

        assertTrue(BOOLEAN.isA(NUMERIC));
        assertTrue(BOOLEAN.isA(NUMBER));
        assertTrue(BOOLEAN.isA(LONG));
        assertTrue(BOOLEAN.isA(BOOLEAN));
        assertTrue(BOOLEAN.isA(DATE));
        assertTrue(BOOLEAN.isA(DOUBLE));

        assertFalse(STRING.isA(NUMBER));
        assertFalse(DATE.isA(IP));

        assertTrue(IP.isA(STRING));
        assertTrue(STRING.isA(IP));
    }
}