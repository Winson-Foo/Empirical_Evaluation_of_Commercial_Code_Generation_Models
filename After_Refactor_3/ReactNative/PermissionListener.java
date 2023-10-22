// PERMISSIONS PACKAGE

package com.facebook.react.modules.core.permissions;

import android.app.Activity;

/**
 * Delegate interface for permission request results.
 */
public interface PermissionListener {

  /**
   * Called when a permission request result is received.
   *
   * @param requestCode The request code passed to requestPermissions()
   * @param permissions The requested permissions
   * @param grantResults The result for each requested permission
   * @return True if the listener can be removed, false otherwise
   */
  boolean onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults);
}

// USAGE PACKAGE

package com.facebook.react.modules.core.usage;

import com.facebook.react.modules.core.permissions.PermissionListener;

public class Usage {

    private final PermissionListener permissionListener;

    public Usage(PermissionListener permissionListener) {
        this.permissionListener = permissionListener;
    }

    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (permissionListener != null) {
            boolean canBeRemoved = permissionListener.onRequestPermissionsResult(requestCode, permissions, grantResults);
            if (canBeRemoved) {
                // Remove the listener from the activity
            }
        }
    }
} 