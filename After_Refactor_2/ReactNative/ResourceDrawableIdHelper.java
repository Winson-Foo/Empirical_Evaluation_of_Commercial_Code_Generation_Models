package com.facebook.react.views.imagehelper;

import android.content.Context;
import android.graphics.drawable.Drawable;
import android.net.Uri;

import androidx.annotation.Nullable;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Locale;

/** Helper class for obtaining information about local images. */
public class ResourceDrawableIdHelper {

    private Map<String, Integer> mResourceDrawableIdMap = new ConcurrentHashMap<>();

    private static final String LOCAL_RESOURCE_SCHEME = "res";

    public ResourceDrawableIdHelper() {}

    public int getResourceDrawableId(Context context, @Nullable String name) {
        if (name == null || name.isEmpty()) {
            return 0;
        }

        name = name.toLowerCase(Locale.ROOT).replace("-", "_");

        // name could be a resource id.
        try {
            return Integer.parseInt(name);
        } catch (NumberFormatException e) {
            // Do nothing.
        }

        if (!mResourceDrawableIdMap.containsKey(name)) {
            int id = context.getResources().getIdentifier(name, "drawable", context.getPackageName());
            mResourceDrawableIdMap.put(name, id);
        }
        return mResourceDrawableIdMap.get(name);
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
}

