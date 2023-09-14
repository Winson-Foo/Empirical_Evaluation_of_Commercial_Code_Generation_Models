package test.java;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import correct_java_programs.SQRT;

public class SQRT_TEST {
    @Test
    public void test_0() throws java.lang.Exception {
        double result = SQRT.calculateSquareRoot((double)2,(double)0.01);
        assertEquals( (double) 1.4166666666666665, result, 0.01);
    }

    @Test
    public void test_1() throws java.lang.Exception {
        double result = SQRT.calculateSquareRoot((double)2,(double)0.5);
        assertEquals( (double) 1.5, result, 0.5);
    }

    @Test
    public void test_2() throws java.lang.Exception {
        double result = SQRT.calculateSquareRoot((double)2,(double)0.3);
        assertEquals( (double) 1.5, result, 0.3);
    }

    @Test
    public void test_3() throws java.lang.Exception {
        double result = SQRT.calculateSquareRoot((double)4,(double)0.2);
        assertEquals( (double) 2, result, 0.2);
    }

    @Test
    public void test_4() throws java.lang.Exception {
        double result = SQRT.calculateSquareRoot((double)27,(double)0.01);
        assertEquals( (double) 5.196164639727311, result, 0.01);
    }

    @Test
    public void test_5() throws java.lang.Exception {
        double result = SQRT.calculateSquareRoot((double)33,(double)0.05);
        assertEquals( (double) 5.744627526262464, result, 0.05);
    }

    @Test
    public void test_6() throws java.lang.Exception {
        double result = SQRT.calculateSquareRoot((double)170,(double)0.03);
        assertEquals( (double) 13.038404876679632, result, 0.03);
    }
}