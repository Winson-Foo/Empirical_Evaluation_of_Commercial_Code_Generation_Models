package test.java;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import correct_java_programs.BITCOUNT;

public class BITCOUNT_TEST {
    @Test
    public void test_0() throws java.lang.Exception {
        int result = BITCOUNT.countSetBits((int)127);
        assertEquals( (int) 7, result);
    }

    @Test
    public void test_1() throws java.lang.Exception {
        int result = BITCOUNT.countSetBits((int)128);
        assertEquals( (int) 1, result);
    }

    @Test
    public void test_2() throws java.lang.Exception {
        int result = BITCOUNT.countSetBits((int)3005);
        assertEquals( (int) 9, result);
    }

    @Test
    public void test_3() throws java.lang.Exception {
        int result = BITCOUNT.countSetBits((int)13);
        assertEquals( (int) 3, result);
    }

    @Test
    public void test_4() throws java.lang.Exception {
        int result = BITCOUNT.countSetBits((int)14);
        assertEquals( (int) 3, result);
    }

    @Test
    public void test_5() throws java.lang.Exception {
        int result = BITCOUNT.countSetBits((int)27);
        assertEquals( (int) 4, result);
    }

    @Test
    public void test_6() throws java.lang.Exception {
        int result = BITCOUNT.countSetBits((int)834);
        assertEquals( (int) 4, result);
    }

    @Test
    public void test_7() throws java.lang.Exception {
        int result = BITCOUNT.countSetBits((int)254);
        assertEquals( (int) 7, result);
    }

    @Test
    public void test_8() throws java.lang.Exception {
        int result = BITCOUNT.countSetBits((int)256);
        assertEquals( (int) 1, result);
    }
}
