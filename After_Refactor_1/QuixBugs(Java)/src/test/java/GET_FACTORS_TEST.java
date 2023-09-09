package test.java;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import java_programs.GET_FACTORS;
import main.java.constant.QuixFixOracleHelper;

public class GET_FACTORS_TEST {
    @Test
    public void test_0() throws java.lang.Exception {
        java.util.ArrayList result = GET_FACTORS.get_factors((int)1);
        String resultFormatted = QuixFixOracleHelper.format(result,true);
        assertEquals("[]", resultFormatted);
    }

    @Test
    public void test_1() throws java.lang.Exception {
        java.util.ArrayList result = GET_FACTORS.get_factors((int)100);
        String resultFormatted = QuixFixOracleHelper.format(result,true);
        assertEquals("[2,2,5,5]", resultFormatted);
    }

    @Test
    public void test_2() throws java.lang.Exception {
        java.util.ArrayList result = GET_FACTORS.get_factors((int)101);
        String resultFormatted = QuixFixOracleHelper.format(result,true);
        assertEquals("[101]", resultFormatted);
    }

    @Test
    public void test_3() throws java.lang.Exception {
        java.util.ArrayList result = GET_FACTORS.get_factors((int)104);
        String resultFormatted = QuixFixOracleHelper.format(result,true);
        assertEquals("[2,2,2,13]", resultFormatted);
    }

    @Test
    public void test_4() throws java.lang.Exception {
        java.util.ArrayList result = GET_FACTORS.get_factors((int)2);
        String resultFormatted = QuixFixOracleHelper.format(result,true);
        assertEquals("[2]", resultFormatted);
    }

    @Test
    public void test_5() throws java.lang.Exception {
        java.util.ArrayList result = GET_FACTORS.get_factors((int)3);
        String resultFormatted = QuixFixOracleHelper.format(result,true);
        assertEquals("[3]", resultFormatted);
    }

    @Test
    public void test_6() throws java.lang.Exception {
        java.util.ArrayList result = GET_FACTORS.get_factors((int)17);
        String resultFormatted = QuixFixOracleHelper.format(result,true);
        assertEquals("[17]", resultFormatted);
    }

    @Test
    public void test_7() throws java.lang.Exception {
        java.util.ArrayList result = GET_FACTORS.get_factors((int)63);
        String resultFormatted = QuixFixOracleHelper.format(result,true);
        assertEquals("[3,3,7]", resultFormatted);
    }

    @Test
    public void test_8() throws java.lang.Exception {
        java.util.ArrayList result = GET_FACTORS.get_factors((int)74);
        String resultFormatted = QuixFixOracleHelper.format(result,true);
        assertEquals("[2,37]", resultFormatted);
    }

    @Test
    public void test_9() throws java.lang.Exception {
        java.util.ArrayList result = GET_FACTORS.get_factors((int)73);
        String resultFormatted = QuixFixOracleHelper.format(result,true);
        assertEquals("[73]", resultFormatted);
    }

    @Test
    public void test_10() throws java.lang.Exception {
        java.util.ArrayList result = GET_FACTORS.get_factors((int)9837);
        String resultFormatted = QuixFixOracleHelper.format(result,true);
        assertEquals("[3,3,1093]", resultFormatted);
    }
}
