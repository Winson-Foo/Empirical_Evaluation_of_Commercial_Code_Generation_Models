// package test.java;
// import org.junit.jupiter.api.Test;
// import static org.junit.jupiter.api.Assertions.assertEquals;
// import correct_java_programs.SIEVE;
// import main.java.constant.QuixFixOracleHelper;

// public class SIEVE_TEST {
//     @Test
//     public void test_0() throws java.lang.Exception {
//         java.util.ArrayList result = SIEVE.sieve((java.lang.Integer)1);
//         String resultFormatted = QuixFixOracleHelper.format(result,true);
//         assertEquals("[]", resultFormatted);
//     }

//     @Test
//     public void test_1() throws java.lang.Exception {
//         java.util.ArrayList result = SIEVE.sieve((java.lang.Integer)2);
//         String resultFormatted = QuixFixOracleHelper.format(result,true);
//         assertEquals("[2]", resultFormatted);
//     }

//     @Test
//     public void test_2() throws java.lang.Exception {
//         java.util.ArrayList result = SIEVE.sieve((java.lang.Integer)4);
//         String resultFormatted = QuixFixOracleHelper.format(result,true);
//         assertEquals("[2,3]", resultFormatted);
//     }

//     @Test
//     public void test_3() throws java.lang.Exception {
//         java.util.ArrayList result = SIEVE.sieve((java.lang.Integer)7);
//         String resultFormatted = QuixFixOracleHelper.format(result,true);
//         assertEquals("[2,3,5,7]", resultFormatted);
//     }

//     @Test
//     public void test_4() throws java.lang.Exception {
//         java.util.ArrayList result = SIEVE.sieve((java.lang.Integer)20);
//         String resultFormatted = QuixFixOracleHelper.format(result,true);
//         assertEquals("[2,3,5,7,11,13,17,19]", resultFormatted);
//     }

//     @Test
//     public void test_5() throws java.lang.Exception {
//         java.util.ArrayList result = SIEVE.sieve((java.lang.Integer)50);
//         String resultFormatted = QuixFixOracleHelper.format(result,true);
//         assertEquals("[2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]", resultFormatted);
//     }
// }