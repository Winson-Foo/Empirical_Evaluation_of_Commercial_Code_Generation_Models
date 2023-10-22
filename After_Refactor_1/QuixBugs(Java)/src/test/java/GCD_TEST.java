package test.java;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import correct_java_programs.GCD;

public class GCD_TEST {
    @Test
    public void test_0() throws java.lang.Exception {
        int result = GCD.calculateGCD((int)13,(int)13);
        assertEquals( (int) 13, result);
    }

    @Test
    public void test_1() throws java.lang.Exception {
        int result = GCD.calculateGCD((int)37,(int)600);
        assertEquals( (int) 1, result);
    }

    @Test
    public void test_2() throws java.lang.Exception {
        int result = GCD.calculateGCD((int)20,(int)100);
        assertEquals( (int) 20, result);
    }

    @Test
    public void test_3() throws java.lang.Exception {
        int result = GCD.calculateGCD((int)624129,(int)2061517);
        assertEquals( (int) 18913, result);
    }

    @Test
    public void test_4() throws java.lang.Exception {
        int result = GCD.calculateGCD((int)3,(int)12);
        assertEquals( (int) 3, result);
    }
}