package org.elasticsearch.xpack.core.watcher;

import java.io.InputStream;

import org.elasticsearch.common.settings.SecureSetting;
import org.elasticsearch.common.settings.Setting;
import org.elasticsearch.license.License;
import org.elasticsearch.license.LicensedFeature;

public final class WatcherField {

    // Watcher settings
    public static final Setting<InputStream> ENCRYPTION_KEY = SecureSetting.secureFile("xpack.watcher.encryption_key", null);

    // Email notification SSL settings
    public static final String EMAIL_NOTIFICATION_SSL_PREFIX = "xpack.notification.email.ssl.";
    public static final String EMAIL_NOTIFICATION_SSL_ENABLED = EMAIL_NOTIFICATION_SSL_PREFIX + "enabled";
    public static final String EMAIL_NOTIFICATION_SSL_KEYSTORE_PATH = EMAIL_NOTIFICATION_SSL_PREFIX + "keystore.path";
    public static final String EMAIL_NOTIFICATION_SSL_KEYSTORE_PASSWORD = EMAIL_NOTIFICATION_SSL_PREFIX + "keystore.password";

    // Licensed features
    public static final LicensedFeature.Momentary WATCHER_FEATURE = LicensedFeature.momentary(
        null,
        "watcher",
        License.OperationMode.STANDARD
    );

    // Private constructor to prevent instantiation
    private WatcherField() {}
}