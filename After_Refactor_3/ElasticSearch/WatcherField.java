/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.core.watcher;

import java.io.InputStream;

import org.elasticsearch.common.settings.SecureSetting;
import org.elasticsearch.common.settings.Setting;
import org.elasticsearch.license.License;
import org.elasticsearch.license.LicensedFeature;

public final class WatcherField {

    // Constants for secure settings
    public static final Setting<InputStream> ENCRYPTION_KEY_SETTING = SecureSetting
            .secureFile("xpack.watcher.encryption_key", null);

    // Constant for email notification SSL prefix
    public static final String EMAIL_NOTIFICATION_SSL_PREFIX = "xpack.notification.email.ssl.";

    // Constant for licensed feature
    public static final LicensedFeature.Momentary WATCHER_FEATURE = LicensedFeature.momentary(null,
            "watcher", License.OperationMode.STANDARD);

    /**
     * Private constructor to prevent this class from being instantiated
     */
    private WatcherField() {
    }
}