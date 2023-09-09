package com.facebook.react.views.imagehelper;

import android.content.Context;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import androidx.annotation.Nullable;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Utility class for getting information about local drawable resources.
 */
public class DrawableResourceUtils {

  private Map<String, Integer> drawableResourceIds;

  private static final String LOCAL_RESOURCE_SCHEME = "res";
  private static volatile DrawableResourceUtils instance;

  private DrawableResourceUtils() {
    drawableResourceIds = new ConcurrentHashMap<>();
  }

  /**
   * Get the singleton instance of this class.
   */
  public static DrawableResourceUtils getInstance() {
    if (instance == null) {
      synchronized (DrawableResourceUtils.class) {
        if (instance == null) {
          instance = new DrawableResourceUtils();
        }
      }
    }
    return instance;
  }

  /**
   * Clear the cached drawable resource ids.
   */
  public void clear() {
    drawableResourceIds.clear();
  }

  /**
   * Get the resource id for the specified drawable name, or 0 if not found.
   */
  public int getDrawableResourceId(Context context, @Nullable String drawableName) {
    if (drawableName == null || drawableName.isEmpty()) {
      return 0;
    }

    // Replace hyphens with underscores and convert to lowercase for consistency.
    drawableName = drawableName.toLowerCase(Locale.ROOT).replace("-", "_");

    // If the drawable name is actually a resource id, return it.
    try {
      return Integer.parseInt(drawableName);
    } catch (NumberFormatException e) {
      // Do nothing.
    }

    // Look up the resource id in the cache or the resources if not found.
    Integer resourceId = drawableResourceIds.get(drawableName);
    if (resourceId == null) {
      resourceId = context.getResources().getIdentifier(drawableName, "drawable", context.getPackageName());
      drawableResourceIds.put(drawableName, resourceId);
    }

    return resourceId != null ? resourceId : 0;
  }

  /**
   * Get the drawable for the specified drawable name, or null if not found.
   */
  public @Nullable Drawable getDrawable(Context context, @Nullable String drawableName) {
    int resourceId = getDrawableResourceId(context, drawableName);
    return resourceId != 0 ? context.getResources().getDrawable(resourceId) : null;
  }

  /**
   * Get the Uri for the specified drawable name, or Uri.EMPTY if not found.
   */
  public Uri getDrawableUri(Context context, @Nullable String drawableName) {
    int resourceId = getDrawableResourceId(context, drawableName);
    return resourceId != 0
        ? new Uri.Builder().scheme(LOCAL_RESOURCE_SCHEME).path(String.valueOf(resourceId)).build()
        : Uri.EMPTY;
  }
}