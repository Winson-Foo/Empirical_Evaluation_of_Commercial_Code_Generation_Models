/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.react.views.text;

import android.graphics.Canvas;
import android.graphics.Paint;
import android.text.style.DynamicDrawableSpan;
import android.text.style.ReplacementSpan;

/**
 * TextInlineViewPlaceholderSpan is a span for inlined views that are inside <Text/> which computes its size based on the input size.
 * It contains no draw logic, just positioning logic.
 */
public class TextInlineViewPlaceholderSpan extends ReplacementSpan implements ReactSpan {

    private int inlineViewTag;
    private int inlineViewWidth;
    private int inlineViewHeight;

    public TextInlineViewPlaceholderSpan(int inlineViewTag, int inlineViewWidth, int inlineViewHeight) {
        this.inlineViewTag = inlineViewTag;
        this.inlineViewWidth = inlineViewWidth;
        this.inlineViewHeight = inlineViewHeight;
    }

    public int getReactTag() {
        return inlineViewTag;
    }

    public int getWidth() {
        return inlineViewWidth;
    }

    public int getHeight() {
        return inlineViewHeight;
    }

    @Override
    public int getSize(Paint paint, CharSequence text, int start, int end, Paint.FontMetricsInt fm) {
        if (fm != null) {
            fm.ascent = -inlineViewHeight;
            fm.descent = 0;
            fm.top = fm.ascent;
            fm.bottom = 0;
        }
        return inlineViewWidth;
    }

    // We don't need to provide a draw implementation as it is empty in this class.
} 