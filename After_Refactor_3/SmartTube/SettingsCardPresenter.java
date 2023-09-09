package com.liskovsoft.smartyoutubetv2.tv.presenter;

import android.content.Context;
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

        initializeColors(context);

        View container = LayoutInflater.from(context).inflate(R.layout.settings_card, null);
        container.setBackgroundColor(mDefaultBackgroundColor);

        TextView textView = initializeTextView(context, container, mDefaultBackgroundColor, mDefaultTextColor);

        setOnFocusChangeListener(textView, context);

        return new ViewHolder(container);
    }

    private void initializeColors(Context context) {
        mDefaultBackgroundColor =
                ContextCompat.getColor(context, Helpers.getThemeAttr(context, R.attr.cardDefaultBackground));
        mDefaultTextColor =
                ContextCompat.getColor(context, R.color.card_default_text);
        mSelectedBackgroundColor =
                ContextCompat.getColor(context, R.color.card_selected_background_white);
        mSelectedTextColor =
                ContextCompat.getColor(context, R.color.card_selected_text_grey);
    }

    private TextView initializeTextView(Context context, View container, int backgroundColor, int textColor) {
        TextView textView = container.findViewById(R.id.settings_title);
        textView.setBackgroundColor(backgroundColor);
        textView.setTextColor(textColor);
        return textView;
    }

    private void setOnFocusChangeListener(TextView textView, Context context) {
        textView.setOnFocusChangeListener((v, hasFocus) -> {
            int backgroundColor = hasFocus ? mSelectedBackgroundColor : mDefaultBackgroundColor;
            int textColor = hasFocus ? mSelectedTextColor : mDefaultTextColor;

            textView.setBackgroundColor(backgroundColor);
            textView.setTextColor(textColor);

            if (hasFocus) {
                enableMarquee(textView, context);
                setTextScrollSpeed(textView, context);
            } else {
                disableMarquee(textView);
            }
        });
    }

    private void enableMarquee(TextView textView, Context context) {
        ViewUtil.enableMarquee(textView);
        ViewUtil.setTextScrollSpeed(textView, MainUIData.instance(context).getCardTextScrollSpeed());
    }

    private void disableMarquee(TextView textView) {
        ViewUtil.disableMarquee(textView);
    }

    private void setTextScrollSpeed(TextView textView, Context context) {
        ViewUtil.setTextScrollSpeed(textView, MainUIData.instance(context).getCardTextScrollSpeed());
    }

    @Override
    public void onBindViewHolder(ViewHolder viewHolder, Object item) {
        SettingsItem settingsItem = (SettingsItem) item;

        TextView textView = viewHolder.view.findViewById(R.id.settings_title);

        textView.setText(settingsItem.title);

        if (settingsItem.imageResId > 0) {
            initializeImageView(settingsItem, viewHolder);
        }
    }

    private void initializeImageView(SettingsItem settingsItem, ViewHolder viewHolder) {
        Context context = viewHolder.view.getContext();

        ImageView imageView = viewHolder.view.findViewById(R.id.settings_image);
        imageView.setImageDrawable(ContextCompat.getDrawable(context, settingsItem.imageResId));
        imageView.setVisibility(View.VISIBLE);
    }

    @Override
    public void onUnbindViewHolder(ViewHolder viewHolder) {
    }
} 