package com.liskovsoft.smartyoutubetv2.common.exoplayer.versions.renderer;

import android.content.Context;
import com.google.android.exoplayer2.DefaultRenderersFactory;
import com.google.android.exoplayer2.Renderer;

import java.util.ArrayList;

public abstract class CustomRenderersFactoryBase extends DefaultRenderersFactory {
    public CustomRenderersFactoryBase(Context context) {
        super(context);
    }

    protected <T extends Renderer> void replaceRenderer(ArrayList<Renderer> renderers, Class<T> rendererClass, T customRenderer) {
        if (renderers != null && customRenderer != null) {
            Renderer originRenderer = null;
            int index = 0;

            for (Renderer renderer : renderers) {
                if (rendererClass.isInstance(renderer)) {
                    originRenderer = renderer;
                    break;
                }
                index++;
            }

            if (originRenderer != null) {
                // replace origin with custom
                renderers.remove(originRenderer);
                renderers.add(index, customRenderer);
            }
        }
    }
} 