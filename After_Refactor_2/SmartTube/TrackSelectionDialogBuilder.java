package com.google.android.exoplayer2.ui;

import android.app.AlertDialog;
import android.app.Dialog;
import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;

import androidx.annotation.Nullable;

import com.google.android.exoplayer2.source.TrackGroupArray;
import com.google.android.exoplayer2.trackselection.DefaultTrackSelector;
import com.google.android.exoplayer2.trackselection.DefaultTrackSelector.SelectionOverride;
import com.google.android.exoplayer2.trackselection.MappingTrackSelector.MappedTrackInfo;
import com.google.android.exoplayer2.trackselection.TrackSelectionUtil;
import com.google.android.exoplayer2.util.Assertions;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class TrackSelectionDialogBuilder {

    public interface DialogCallback {
        void onTracksSelected(boolean isDisabled, List<SelectionOverride> overrides);
    }

    private final Context context;
    private final CharSequence title;
    private final MappedTrackInfo mappedTrackInfo;
    private final int rendererIndex;
    private final DialogCallback callback;

    public TrackSelectionDialogBuilder(
            Context context,
            CharSequence title,
            MappedTrackInfo mappedTrackInfo,
            int rendererIndex,
            DialogCallback callback) {
        this.context = context;
        this.title = title;
        this.mappedTrackInfo = mappedTrackInfo;
        this.rendererIndex = rendererIndex;
        this.callback = callback;
    }

    public TrackSelectionDialogBuilder(Context context, CharSequence title, DefaultTrackSelector trackSelector, int rendererIndex) {
        this.context = context;
        this.title = title;
        this.mappedTrackInfo = Assertions.checkNotNull(trackSelector.getCurrentMappedTrackInfo());
        this.rendererIndex = rendererIndex;
        DefaultTrackSelector.Parameters selectionParameters = trackSelector.getParameters();
        boolean isDisabled = selectionParameters.getRendererDisabled(rendererIndex);
        List<SelectionOverride> overrides = new ArrayList<>();
        SelectionOverride override = selectionParameters.getSelectionOverride(rendererIndex, mappedTrackInfo.getTrackGroups(rendererIndex));
        if (override != null) {
            overrides.add(override);
        }

        this.callback = new DialogCallback() {
            @Override
            public void onTracksSelected(boolean isDisabled, List<SelectionOverride> newOverrides) {
                trackSelector.setParameters(
                    TrackSelectionUtil.updateParametersWithOverride(
                            selectionParameters,
                            rendererIndex,
                            mappedTrackInfo.getTrackGroups(rendererIndex),
                            isDisabled,
                            newOverrides.isEmpty() ? null : newOverrides.get(0)));
            }
        };
    }

    public TrackSelectionDialogBuilder setIsDisabled(boolean isDisabled) {
        this.isDisabled = isDisabled;
        return this;
    }

    public TrackSelectionDialogBuilder setOverride(@Nullable SelectionOverride override) {
        return setOverrides(override == null ? Collections.emptyList() : Collections.singletonList(override));
    }

    public TrackSelectionDialogBuilder setOverrides(List<SelectionOverride> overrides) {
        this.overrides = new ArrayList<>(overrides);
        return this;
    }

    public TrackSelectionDialogBuilder setAllowAdaptiveSelections(boolean enabled) {
        this.isAdaptiveSelectionEnabled = enabled;
        return this;
    }

    public TrackSelectionDialogBuilder setAllowMultipleOverrides(boolean multipleOverrides) {
        this.multipleOverridesEnabled = multipleOverrides;
        return this;
    }

    public TrackSelectionDialogBuilder setShowDisableOption(boolean showDisableOption) {
        this.showDisableOption = showDisableOption;
        return this;
    }

    public TrackSelectionDialogBuilder setTrackNameProvider(@Nullable TrackNameProvider trackNameProvider) {
        this.trackNameProvider = trackNameProvider;
        return this;
    }

    public AlertDialog build() {
        AlertDialog.Builder builder = new AlertDialog.Builder(context);
        LayoutInflater inflater = LayoutInflater.from(builder.getContext());
        View view = inflater.inflate(R.layout.exo_track_selection_dialog, null);

        TrackSelectionView selectionView = view.findViewById(R.id.exo_track_selection_view);
        selectionView.setAllowMultipleOverrides(multipleOverridesEnabled);
        selectionView.setAllowAdaptiveSelections(isAdaptiveSelectionEnabled);
        selectionView.setShowDisableOption(showDisableOption);
        if (trackNameProvider != null) {
            selectionView.setTrackNameProvider(trackNameProvider);
        }
        selectionView.init(mappedTrackInfo, rendererIndex, isDisabled, overrides, null);
        Dialog.OnClickListener okClickListener = (dialog, which) -> {
            boolean isDisabled = selectionView.getIsDisabled();
            List<SelectionOverride> overrides = selectionView.getOverrides();
            callback.onTracksSelected(isDisabled, overrides);
        };

        return builder
                .setTitle(title)
                .setView(view)
                .setPositiveButton(android.R.string.ok, okClickListener)
                .setNegativeButton(android.R.string.cancel, null)
                .create();
    }

    private boolean isDisabled;
    private List<SelectionOverride> overrides;
    private boolean isAdaptiveSelectionEnabled;
    private boolean multipleOverridesEnabled;
    private boolean showDisableOption;
    private TrackNameProvider trackNameProvider;

}

