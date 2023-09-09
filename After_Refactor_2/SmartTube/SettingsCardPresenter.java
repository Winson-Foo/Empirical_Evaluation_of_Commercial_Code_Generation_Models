package com.liskovsoft.smartyoutubetv2.tv.presenter;

import android.content.Context;
import android.graphics.drawable.Drawable;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.core.content.ContextCompat;
import androidx.leanback.widget.Presenter;

import com.liskovsoft.sharedutils.helpers.Helpers;
import com.liskovsoft.smartyoutubetv2.common.app.models.data.SettingsItem;
import com.liskovsoft.smartyoutubetv2.common.prefs.MainUIData;
import com.liskovsoft.smartyoutubetv2.tv.R;
import com.liskovsoft.smartyoutubetv2.tv.util.ViewUtil;

public class SettingsCardPresenter extends Presenter {

    private int mDefaultBackgroundColor;
    private int mDefaultTextColor;
    private int mSelectedBackgroundColor;
    private int mSelectedTextColor;

    @Override
    public ViewHolder onCreateViewHolder(ViewGroup parent) {
        Context context = parent.getContext();
        initColors(context);

        View container = LayoutInflater.from(context).inflate(R.layout.settings_card, null);
        container.setBackgroundColor(mDefaultBackgroundColor);

        TextView textView = container.findViewById(R.id.settings_title);
        textView.setBackgroundColor(mDefaultBackgroundColor);
        textView.setTextColor(mDefaultTextColor);

        container.setOnFocusChangeListener((v, hasFocus) -> updateColorsOnFocusChange(hasFocus, textView, context));

        return new ViewHolder(container);
    }

    private void initColors(Context context) {
        mDefaultBackgroundColor =
                ContextCompat.getColor(context, Helpers.getThemeAttr(context, R.attr.cardDefaultBackground));
        mDefaultTextColor =
                ContextCompat.getColor(context, R.color.card_default_text);
        mSelectedBackgroundColor =
                ContextCompat.getColor(context, R.color.card_selected_background_white);
        mSelectedTextColor =
                ContextCompat.getColor(context, R.color.card_selected_text_grey);
    }

    private void updateColorsOnFocusChange(boolean hasFocus, TextView textView, Context context) {
        int backgroundColor = hasFocus ? mSelectedBackgroundColor : mDefaultBackgroundColor;
        int textColor = hasFocus ? mSelectedTextColor : mDefaultTextColor;

        textView.setBackgroundColor(backgroundColor);
        textView.setTextColor(textColor);

        if (hasFocus) {
            ViewUtil.enableMarquee(textView);
            ViewUtil.setTextScrollSpeed(textView, MainUIData.instance(context).getCardTextScrollSpeed());
        } else {
            ViewUtil.disableMarquee(textView);
        }
    }

    @Override
    public void onBindViewHolder(ViewHolder viewHolder, Object item) {
        SettingsItem settingsItem = (SettingsItem) item;
        TextView textView = viewHolder.view.findViewById(R.id.settings_title);

        textView.setText(settingsItem.title);
        updateImage(viewHolder.view.getContext(), settingsItem.imageResId, viewHolder.view.findViewById(R.id.settings_image));
    }

    private void updateImage(Context context, int imageResId, ImageView imageView) {
        if (imageResId > 0) {
            Drawable drawable = ContextCompat.getDrawable(context, imageResId);
            if (drawable != null) {
                imageView.setImageDrawable(drawable);
                imageView.setVisibility(View.VISIBLE);
            }
        } else {
            imageView.setVisibility(View.GONE);
        }
    }

    @Override
    public void onUnbindViewHolder(ViewHolder viewHolder) {
    }
} 