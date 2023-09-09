package com.facebook.react.views.imagehelper;

import android.content.Context;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import androidx.annotation.Nullable;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import javax.annotation.concurrent.ThreadSafe;

/** Helper class for obtaining information about local images. */
@ThreadSafe
public class ResourceDrawableIdHelper {

  private final Map<String, Integer> mResourceDrawableIdMap = new HashMap<>();
  private static final String LOCAL_RESOURCE_SCHEME = "res";
  private static final ResourceDrawableIdHelper sInstance = new ResourceDrawableIdHelper();

  private ResourceDrawableIdHelper() {}

  public static ResourceDrawableIdHelper getInstance() {
    return sInstance;
  }

  public synchronized void clear() {
    mResourceDrawableIdMap.clear();
  }

  public int getResourceDrawableId(Context context, @Nullable String name) {
    if (isEmpty(name)) {
      return 0;
    }

    name = normalize(name);

    // name could be a resource id.
    try {
      return Integer.parseInt(name);
    } catch (NumberFormatException e) {
      // Do nothing.
    }

    synchronized (this) {
      if (mResourceDrawableIdMap.containsKey(name)) {
        return mResourceDrawableIdMap.get(name);
      }
      int id = context.getResources().getIdentifier(name, "drawable", context.getPackageName());
      mResourceDrawableIdMap.put(name, id);
      return id;
    }
  }

  public @Nullable Drawable getResourceDrawable(Context context, @Nullable String name) {
    int resId = getResourceDrawableId(context, name);
    return resId > 0 ? context.getResources().getDrawable(resId) : null;
  }

  public Uri getResourceDrawableUri(Context context, @Nullable String name) {
    int resId = getResourceDrawableId(context, name);
    return resId > 0
        ? new Uri.Builder().scheme(LOCAL_RESOURCE_SCHEME).path(String.valueOf(resId)).build()
        : Uri.EMPTY;
  }

  private boolean isEmpty(@Nullable String str) {
    return str == null || str.isEmpty();
  }

  private String normalize(@Nullable String str) {
    return isEmpty(str) ? "" : str.toLowerCase(Locale.ROOT).replace("-", "_");
  }
}