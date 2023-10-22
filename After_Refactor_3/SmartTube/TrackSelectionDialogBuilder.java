package com.google.android.exoplayer2.ui;

import android.app.AlertDialog;
import android.app.Dialog;
import android.content.Context;
import androidx.annotation.Nullable;
import android.view.LayoutInflater;
import android.view.View;

import com.google.android.exoplayer2.source.TrackGroupArray;
import com.google.android.exoplayer2.trackselection.DefaultTrackSelector;
import com.google.android.exoplayer2.trackselection.DefaultTrackSelector.SelectionOverride;
import com.google.android.exoplayer2.trackselection.MappingTrackSelector.MappedTrackInfo;
import com.google.android.exoplayer2.trackselection.TrackSelectionUtil;
import com.google.android.exoplayer2.util.Assertions;

import java.util.Collections;
import java.util.List;

/** Builder for a dialog with a {@link TrackSelectionView}. */
public final class TrackSelectionDialogBuilder {

  public static final int BUTTON_OK = android.R.string.ok;
  public static final int BUTTON_CANCEL = android.R.string.cancel;

  private final Context context;
  private final CharSequence title;
  private final MappedTrackInfo mappedTrackInfo;
  private final int rendererIndex;
  private final DialogCallback callback;

  private boolean allowAdaptiveSelections;
  private boolean allowMultipleOverrides;
  private boolean showDisableOption;
  @Nullable private TrackNameProvider trackNameProvider;
  private boolean isDisabled;
  private List<SelectionOverride> overrides;

  /**
   * Creates a builder for a track selection dialog.
   *
   * @param context          The context of the dialog.
   * @param title            The title of the dialog.
   * @param mappedTrackInfo  The {@link MappedTrackInfo} containing the track information.
   * @param rendererIndex    The renderer index in the {@code mappedTrackInfo} for which the track
   *                         selection is shown.
   * @param callback         The {@link DialogCallback} invoked when a track selection has been made.
   */
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
    this.overrides = Collections.emptyList();
  }

  /**
   * Creates a builder for a track selection dialog which automatically updates a {@link
   * DefaultTrackSelector}.
   *
   * @param context       The context of the dialog.
   * @param title         The title of the dialog.
   * @param trackSelector A {@link DefaultTrackSelector} whose current selection is used to set up
   *                      the dialog and which is updated when new tracks are selected in the dialog.
   * @param rendererIndex The renderer index in the {@code trackSelector} for which the track
   *                      selection is shown.
   */
  public TrackSelectionDialogBuilder(
          Context context, CharSequence title, DefaultTrackSelector trackSelector, int rendererIndex) {
    this.context = context;
    this.title = title;
    this.mappedTrackInfo = Assertions.checkNotNull(trackSelector.getCurrentMappedTrackInfo());
    this.rendererIndex = rendererIndex;

    TrackGroupArray rendererTrackGroups = mappedTrackInfo.getTrackGroups(rendererIndex);
    DefaultTrackSelector.Parameters selectionParameters = trackSelector.getParameters();
    isDisabled = selectionParameters.getRendererDisabled(rendererIndex);
    SelectionOverride override =
            selectionParameters.getSelectionOverride(rendererIndex, rendererTrackGroups);
    overrides = override == null ? Collections.emptyList() : Collections.singletonList(override);

    this.callback = (newIsDisabled, newOverrides) -> {
      DefaultTrackSelector.Parameters parameters = TrackSelectionUtil.updateParametersWithOverride(
              selectionParameters,
              rendererIndex,
              rendererTrackGroups,
              newIsDisabled,
              newOverrides.isEmpty() ? null : newOverrides.get(0)
      );
      trackSelector.setParameters(parameters);
    };
  }

  public TrackSelectionDialogBuilder setAllowAdaptiveSelections(boolean allowAdaptiveSelections) {
    this.allowAdaptiveSelections = allowAdaptiveSelections;
    return this;
  }

  public TrackSelectionDialogBuilder setAllowMultipleOverrides(boolean allowMultipleOverrides) {
    this.allowMultipleOverrides = allowMultipleOverrides;
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

  public TrackSelectionDialogBuilder setIsDisabled(boolean isDisabled) {
    this.isDisabled = isDisabled;
    return this;
  }

  public TrackSelectionDialogBuilder setOverride(@Nullable SelectionOverride override) {
    return setOverrides(
            override == null ? Collections.emptyList() : Collections.singletonList(override));
  }

  public TrackSelectionDialogBuilder setOverrides(List<SelectionOverride> overrides) {
    this.overrides = overrides;
    return this;
  }

  public AlertDialog build() {
    AlertDialog.Builder builder = new AlertDialog.Builder(context);

    View dialogView = LayoutInflater.from(builder.getContext()).inflate(R.layout.exo_track_selection_dialog, null);

    initSelectionView(dialogView);

    Dialog.OnClickListener okClickListener = (dialog, which) ->
            callback.onTracksSelected(getSelectionView().getIsDisabled(), getSelectionView().getOverrides());

    return builder
            .setTitle(title)
            .setView(dialogView)
            .setPositiveButton(BUTTON_OK, okClickListener)
            .setNegativeButton(BUTTON_CANCEL, null)
            .create();
  }

  private void initSelectionView(View dialogView) {
    TrackSelectionView selectionView = dialogView.findViewById(R.id.exo_track_selection_view);
    selectionView.setAllowMultipleOverrides(allowMultipleOverrides);
    selectionView.setAllowAdaptiveSelections(allowAdaptiveSelections);
    selectionView.setShowDisableOption(showDisableOption);
    if (trackNameProvider != null) {
      selectionView.setTrackNameProvider(trackNameProvider);
    }
    selectionView.init(mappedTrackInfo, rendererIndex, isDisabled, overrides, null);
    setSelectionView(selectionView);
  }

  /** Callback which is invoked when a track selection has been made. */
  public interface DialogCallback {

    /**
     * Called when tracks are selected.
     *
     * @param isDisabled Whether the renderer is disabled.
     * @param overrides  List of selected track selection overrides for the renderer.
     */
    void onTracksSelected(boolean isDisabled, List<SelectionOverride> overrides);
  }

  private TrackSelectionView selectionView;

  public TrackSelectionView getSelectionView() {
    return selectionView;
  }

  public void setSelectionView(TrackSelectionView selectionView) {
    this.selectionView = selectionView;
  }
} 