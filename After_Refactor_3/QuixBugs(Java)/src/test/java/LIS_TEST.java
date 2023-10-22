package test.java;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import correct_java_programs.LIS;

public class LIS_TEST {
    @Test
    public void test_0() throws java.lang.Exception {
        int result = LIS.calculateLis(new int[]{4,1,5,3,7,6,2});
        assertEquals( (int) 3, result);
    }

    @Test
    public void test_1() throws java.lang.Exception {
        int result = LIS.calculateLis(new int[]{10,22,9,33,21,50,41,60,80});
        assertEquals( (int) 6, result);
    }

    @Test
    public void test_2() throws java.lang.Exception {
        int result = LIS.calculateLis(new int[]{7,10,9,2,3,8,1});
        assertEquals( (int) 3, result);
    }

    @Test
    public void test_3() throws java.lang.Exception {
        int result = LIS.calculateLis(new int[]{9,11,2,13,7,15});
        assertEquals( (int) 4, result);
    }
}