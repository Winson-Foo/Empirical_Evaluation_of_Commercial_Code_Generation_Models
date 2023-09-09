package test.java;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import java_programs.PASCAL;
import main.java.constant.QuixFixOracleHelper;

public class PASCAL_TEST {
    @Test
    public void test_0() throws java.lang.Exception {
        java.util.ArrayList result = PASCAL.pascal((int)1);
        String resultFormatted = QuixFixOracleHelper.format(result,true);
        assertEquals("[[1]]", resultFormatted);
    }

    @Test
    public void test_1() throws java.lang.Exception {
        java.util.ArrayList result = PASCAL.pascal((int)2);
        String resultFormatted = QuixFixOracleHelper.format(result,true);
        assertEquals("[[1],[1,1]]", resultFormatted);
    }

    @Test
    public void test_2() throws java.lang.Exception {
        java.util.ArrayList result = PASCAL.pascal((int)3);
        String resultFormatted = QuixFixOracleHelper.format(result,true);
        assertEquals("[[1],[1,1],[1,2,1]]", resultFormatted);
    }

    @Test
    public void test_3() throws java.lang.Exception {
        java.util.ArrayList result = PASCAL.pascal((int)4);
        String resultFormatted = QuixFixOracleHelper.format(result,true);
        assertEquals("[[1],[1,1],[1,2,1],[1,3,3,1]]", resultFormatted);
    }

    @Test
    public void test_4() throws java.lang.Exception {
        java.util.ArrayList result = PASCAL.pascal((int)5);
        String resultFormatted = QuixFixOracleHelper.format(result,true);
        assertEquals("[[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]", resultFormatted);
    }
}