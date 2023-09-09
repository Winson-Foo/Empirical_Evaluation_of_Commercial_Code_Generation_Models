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
    private final int defaultBackgroundColor;
    private final int defaultTextColor;
    private final int selectedBackgroundColor;
    private final int selectedTextColor;

    public SettingsCardPresenter(Context context) {
        defaultBackgroundColor = ContextCompat.getColor(context, Helpers.getThemeAttr(context, R.attr.cardDefaultBackground));
        defaultTextColor = ContextCompat.getColor(context, R.color.card_default_text);
        selectedBackgroundColor = ContextCompat.getColor(context, R.color.card_selected_background_white);
        selectedTextColor = ContextCompat.getColor(context, R.color.card_selected_text_grey);
    }

    @Override
    public SettingsViewHolder onCreateViewHolder(ViewGroup parent) {
        View container = LayoutInflater.from(parent.getContext()).inflate(R.layout.settings_card, null);
        container.setBackgroundColor(defaultBackgroundColor);

        return new SettingsViewHolder(container);
    }

    @Override
    public void onBindViewHolder(SettingsViewHolder viewHolder, Object item) {
        SettingsItem settingsItem = (SettingsItem) item;

        bindView(viewHolder, settingsItem);
        bindFocus(viewHolder, settingsItem);
    }

    @Override
    public void onUnbindViewHolder(SettingsViewHolder viewHolder) {
    }

    private void bindView(SettingsViewHolder viewHolder, SettingsItem settingsItem) {
        TextView textView = viewHolder.view.findViewById(R.id.settings_title);
        textView.setText(settingsItem.title);

        if (settingsItem.imageResId > 0) {
            ImageView imageView = viewHolder.view.findViewById(R.id.settings_image);
            imageView.setImageDrawable(ContextCompat.getDrawable(viewHolder.view.getContext(), settingsItem.imageResId));
            imageView.setVisibility(View.VISIBLE);
        }
    }

    private void bindFocus(SettingsViewHolder viewHolder, SettingsItem settingsItem) {
        TextView textView = viewHolder.view.findViewById(R.id.settings_title);

        viewHolder.view.setOnFocusChangeListener((v, hasFocus) -> {
            int backgroundColor = hasFocus ? selectedBackgroundColor : defaultBackgroundColor;
            int textColor = hasFocus ? selectedTextColor : defaultTextColor;
            
            textView.setBackgroundColor(backgroundColor);
            textView.setTextColor(textColor);

            if (hasFocus) {
                ViewUtil.enableMarquee(textView);
                ViewUtil.setTextScrollSpeed(textView, MainUIData.instance(viewHolder.view.getContext()).getCardTextScrollSpeed());
            } else {
                ViewUtil.disableMarquee(textView);
            }
        });
    }

    public static class SettingsViewHolder extends Presenter.ViewHolder {
        public SettingsViewHolder(View view) {
            super(view);
        }
    }
} 