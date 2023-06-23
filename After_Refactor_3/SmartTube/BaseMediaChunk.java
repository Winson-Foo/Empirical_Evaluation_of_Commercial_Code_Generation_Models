package com.google.android.exoplayer2.source.chunk;

import com.google.android.exoplayer2.C;
import com.google.android.exoplayer2.Format;
import com.google.android.exoplayer2.upstream.DataSource;
import com.google.android.exoplayer2.upstream.DataSpec;

/**
 * A base implementation of {@link MediaChunk} that outputs to a {@link BaseMediaChunkOutput}.
 * Provides methods for initializing the chunk and retrieving output information.
 */
public abstract class BaseMediaChunk extends MediaChunk {

  /** The default value for clipped start and end times. */
  public static final long DEFAULT_TIME = C.TIME_UNSET;

  /** The time from which output will begin. */
  private final long clippedStartTimeUs;
  /** The time from which output will end. */
  private final long clippedEndTimeUs;

  private BaseMediaChunkOutput output;
  private int[] firstSampleIndices;

  /**
   * Creates a new instance of {@link BaseMediaChunk}.
   *
   * @param dataSource The source from which the data should be loaded.
   * @param dataSpec Defines the data to be loaded.
   * @param trackFormat The format of the track.
   * @param trackSelectionReason The reason for selecting this track.
   * @param trackSelectionData Additional data for custom track selection.
   * @param startTimeUs The start time of the chunk.
   * @param endTimeUs The end time of the chunk.
   * @param clippedStartTimeUs The time in the chunk from which output will begin. Pass {@link
   *     #DEFAULT_TIME} if output should begin from the start.
   * @param clippedEndTimeUs The time in the chunk from which output will end. Pass {@link
   *     #DEFAULT_TIME} if output should end at the end.
   * @param chunkIndex The index of the chunk, or {@link C#INDEX_UNSET} if unknown.
   */
  public BaseMediaChunk(
      DataSource dataSource,
      DataSpec dataSpec,
      Format trackFormat,
      int trackSelectionReason,
      Object trackSelectionData,
      long startTimeUs,
      long endTimeUs,
      long clippedStartTimeUs,
      long clippedEndTimeUs,
      long chunkIndex) {
    super(dataSource, dataSpec, trackFormat, trackSelectionReason, trackSelectionData, startTimeUs,
        endTimeUs, chunkIndex);

    this.clippedStartTimeUs = clippedStartTimeUs;
    this.clippedEndTimeUs = clippedEndTimeUs;
  }

  /**
   * Initializes the chunk for loading.
   *
   * @param output The output that will receive the loaded media samples.
   */
  public void init(BaseMediaChunkOutput output) {
    this.output = output;
    firstSampleIndices = output.getWriteIndices();
  }

  /**
   * Returns the index of the first sample in the specified track of the output that will originate
   * from this chunk.
   *
   * @param trackIndex The index of the track.
   * @return The index of the first sample.
   */
  public final int getFirstSampleIndex(int trackIndex) {
    return firstSampleIndices[trackIndex];
  }

  /**
   * Returns the {@link BaseMediaChunkOutput} that was passed to {@link #init(BaseMediaChunkOutput)}.
   *
   * @return The {@link BaseMediaChunkOutput}.
   */
  protected final BaseMediaChunkOutput getOutput() {
    return output;
  }

  /**
   * Returns the clipped start time of the chunk.
   *
   * @return The clipped start time of the chunk.
   */
  protected final long getClippedStartTimeUs() {
    return clippedStartTimeUs;
  }

  /**
   * Returns the clipped end time of the chunk.
   *
   * @return The clipped end time of the chunk.
   */
  protected final long getClippedEndTimeUs() {
    return clippedEndTimeUs;
  }

} 