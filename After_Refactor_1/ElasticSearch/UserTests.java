package org.elasticsearch.xpack.core.security.user;

import org.elasticsearch.test.ESTestCase;

import java.util.Map;

import static org.hamcrest.Matchers.is;

/**
 * Unit tests for the User class.
 */
public class UserTests extends ESTestCase {

    private static final String USERNAME = "u1";
    private static final String[] ROLES = { "r1", "r2" };
    private static final String FULL_NAME = "user1";
    private static final String EMAIL = "user1@domain.com";
    private static final Map<String, String> METADATA = Map.of("key", "val");

    /**
     * Tests that a user's string representation is correctly formatted.
     */
    public void testUserToString() {
        testUserToStringSingleRole();
        testUserToStringMultipleRoles();
    }

    /**
     * Tests that a user with a single role is correctly formatted.
     */
    private void testUserToStringSingleRole() {
        User user = new User(USERNAME, ROLES[0]);
        String expected = String.format("User[username=%s,roles=[%s],fullName=null,email=null,metadata={}]",
            USERNAME, ROLES[0]);
        assertThat(user.toString(), is(expected));
    }

    /**
     * Tests that a user with multiple roles is correctly formatted.
     */
    private void testUserToStringMultipleRoles() {
        User user = new User(USERNAME, ROLES, FULL_NAME, EMAIL, METADATA, true);
        String expected = String.format("User[username=%s,roles=[%s,%s],fullName=%s,email=%s,metadata={key=val}]",
            USERNAME, ROLES[0], ROLES[1], FULL_NAME, EMAIL);
        assertThat(user.toString(), is(expected));
    }
}