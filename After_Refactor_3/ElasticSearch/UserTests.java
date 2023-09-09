package org.elasticsearch.xpack.core.security.user;

import org.elasticsearch.test.ESTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import static org.hamcrest.Matchers.is;
import static org.junit.Assert.assertThat;

import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

@RunWith(Parameterized.class)
public class UserTests extends ESTestCase {

    private final String username;
    private final String[] roles;
    private final String fullName;
    private final String email;
    private final Map<String, Object> metadata;
    private final boolean enabled;

    public UserTests(String username, String[] roles, String fullName, String email,
                     Map<String, Object> metadata, boolean enabled) {
        this.username = username;
        this.roles = roles;
        this.fullName = fullName;
        this.email = email;
        this.metadata = metadata;
        this.enabled = enabled;
    }

    @Parameterized.Parameters
    public static Collection testParams() {
        return Arrays.asList(new Object[][] {
            { "u1", new String[] { "r1" }, null, null, Map.of(), false },
            { "u2", new String[] { "r1", "r2" }, "user1", "user1@domain.com",
              Map.of("key", "val"), true}
        });
    }

    @Test
    public void testToString() {
        User user = new User(username, roles, fullName, email, metadata, enabled);
        String expected = "User[username=" + username + ",roles=" +
                          Arrays.toString(roles) + ",fullName=" + fullName +
                          ",email=" + email + ",metadata=" + metadata + "]";
        assertThat(user.toString(), is(expected));
    }
}