package org.elasticsearch.xpack.core.watcher;

import org.elasticsearch.common.settings.SecureSetting;
import org.elasticsearch.common.settings.Setting;
import org.elasticsearch.license.License;
import org.elasticsearch.license.LicensedFeature;

import java.io.InputStream;

/**
 * Constants and settings used by Watcher.
 */
public final class WatcherField {

    /**
     * The setting for the encryption key used by Watcher.
     */
    public static final Setting<InputStream> ENCRYPTION_KEY_SETTING = SecureSetting.secureFile("xpack.watcher.encryption_key", null);

    /**
     * The prefix for SSL related settings used by email notifications.
     */
    public static final String EMAIL_NOTIFICATION_SSL_PREFIX = "xpack.notification.email.ssl.";

    /**
     * The momentary licensed feature for Watcher.
     */
    public static final LicensedFeature.Momentary WATCHER_FEATURE = LicensedFeature.momentary(
        null,
        "watcher",
        License.OperationMode.STANDARD
    );

    private WatcherField() {}

}