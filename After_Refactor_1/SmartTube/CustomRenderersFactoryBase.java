package com.liskovsoft.smartyoutubetv2.common.exoplayer.versions.renderer;

import android.content.Context;
import com.google.android.exoplayer2.DefaultRenderersFactory;
import com.google.android.exoplayer2.Renderer;
import com.google.android.exoplayer2.audio.MediaCodecAudioRenderer;
import com.google.android.exoplayer2.video.MediaCodecVideoRenderer;

import java.util.ArrayList;

public abstract class CustomRenderersFactoryBase extends DefaultRenderersFactory {

    public CustomRenderersFactoryBase(Context context) {
        super(context);
    }

    protected void replaceRenderer(ArrayList<Renderer> renderers, Renderer renderer, Class<?> clazz) {
        if (renderers != null && renderer != null && clazz != null) {
            Renderer originRenderer = null;
            int index = 0;

            for (Renderer r : renderers) {
                if (clazz.isInstance(r)) {
                    originRenderer = r;
                    break;
                }
                index++;
            }

            if (originRenderer != null) {
                // replace origin with custom
                renderers.remove(originRenderer);
                renderers.add(index, renderer);
            }
        }
    }

    protected void replaceVideoRenderer(ArrayList<Renderer> renderers, MediaCodecVideoRenderer videoRenderer) {
        replaceRenderer(renderers, videoRenderer, MediaCodecVideoRenderer.class);
    }

    protected void replaceAudioRenderer(ArrayList<Renderer> renderers, MediaCodecAudioRenderer audioRenderer) {
        replaceRenderer(renderers, audioRenderer, MediaCodecAudioRenderer.class);
    }
}