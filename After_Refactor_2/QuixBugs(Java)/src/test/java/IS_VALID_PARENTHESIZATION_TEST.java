package test.java;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import java_programs.IS_VALID_PARENTHESIZATION;
import main.java.constant.QuixFixOracleHelper;

public class IS_VALID_PARENTHESIZATION_TEST {
    @Test
    public void test_0() throws java.lang.Exception {
        java.lang.Boolean result = IS_VALID_PARENTHESIZATION.is_valid_parenthesization((java.lang.String)"((()()))()");
        String resultFormatted = QuixFixOracleHelper.format(result,true);
        assertEquals("true", resultFormatted);
    }

    @Test
    public void test_1() throws java.lang.Exception {
        java.lang.Boolean result = IS_VALID_PARENTHESIZATION.is_valid_parenthesization((java.lang.String)")()(");
        String resultFormatted = QuixFixOracleHelper.format(result,true);
        assertEquals("false", resultFormatted);
    }

    @Test
    public void test_2() throws java.lang.Exception {
        java.lang.Boolean result = IS_VALID_PARENTHESIZATION.is_valid_parenthesization((java.lang.String)"((");
        String resultFormatted = QuixFixOracleHelper.format(result,true);
        assertEquals("false", resultFormatted);
    }
}