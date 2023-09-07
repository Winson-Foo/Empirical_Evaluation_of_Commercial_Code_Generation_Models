package test.java;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import java_programs.LEVENSHTEIN;

public class LEVENSHTEIN_TEST {
    @Test
    public void test_0() throws java.lang.Exception {
        int result = LEVENSHTEIN.levenshtein((java.lang.String)"electron",(java.lang.String)"neutron");
        assertEquals( (int) 3, result);
    }

    @Test
    public void test_1() throws java.lang.Exception {
        int result = LEVENSHTEIN.levenshtein((java.lang.String)"kitten",(java.lang.String)"sitting");
        assertEquals( (int) 3, result);
    }

    @Test
    public void test_2() throws java.lang.Exception {
        int result = LEVENSHTEIN.levenshtein((java.lang.String)"rosettacode",(java.lang.String)"raisethysword");
        assertEquals( (int) 8, result);
    }

    @Test
    public void test_3() throws java.lang.Exception {
        int result = LEVENSHTEIN.levenshtein((java.lang.String)"abcdefg",(java.lang.String)"gabcdef");
        assertEquals( (int) 2, result);
    }

    @Test
    public void test_4() throws java.lang.Exception {
        int result = LEVENSHTEIN.levenshtein((java.lang.String)"",(java.lang.String)"");
        assertEquals( (int) 0, result);
    }

    @Test
    public void test_5() throws java.lang.Exception {
        int result = LEVENSHTEIN.levenshtein((java.lang.String)"hello",(java.lang.String)"olleh");
        assertEquals( (int) 4, result);
    }
}