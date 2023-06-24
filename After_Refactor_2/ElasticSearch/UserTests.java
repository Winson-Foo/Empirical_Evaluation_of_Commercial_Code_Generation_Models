package org.elasticsearch.xpack.core.security.user;

import org.elasticsearch.test.ESTestCase;

import java.util.Map;

import static org.hamcrest.Matchers.is;

public class UserTests extends ESTestCase {

    private static final String USERNAME = "u1";
    private static final String[] ROLES = {"r1", "r2"};
    private static final String FULL_NAME = "user1";
    private static final String EMAIL = "user1@domain.com";
    private static final Map<String, String> METADATA = Map.of("key", "val");
    private static final boolean ENABLED = true;

    private User createUser(String username, String[] roles, String fullName, String email, Map<String, String> metadata, boolean enabled) {
        return new User(username, roles, fullName, email, metadata, enabled);
    }

    public void testUserToString() {
        User user = createUser(USERNAME, new String[] {"r1"}, null, null, null, false);
        assertThat(user.toString(), is("User[username=" + USERNAME + ",roles=[r1],fullName=null,email=null,metadata={}]"));

        user = createUser(USERNAME, ROLES, FULL_NAME, EMAIL, METADATA, ENABLED);
        assertThat(user.toString(), is("User[username=" + USERNAME + ",roles=[r1,r2],fullName=" + FULL_NAME + ",email=" + EMAIL + ",metadata={key=val}]"));
    }
}