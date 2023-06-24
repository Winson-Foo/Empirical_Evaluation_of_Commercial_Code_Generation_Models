package com.google.android.exoplayer2.source.chunk;

import com.google.android.exoplayer2.C;
import com.google.android.exoplayer2.Format;
import com.google.android.exoplayer2.upstream.DataSource;
import com.google.android.exoplayer2.upstream.DataSpec;

/**
 * A base implementation of {@link MediaChunk} that outputs to a {@link BaseMediaChunkOutput}.
 */
public abstract class BaseMediaChunk extends MediaChunk {

  private final long clippedStartTimeUs;
  private final long clippedEndTimeUs;

  private BaseMediaChunkOutput output;
  private int[] firstSampleIndices;

  /**
   * Creates a new BaseMediaChunk.
   *
   * @param dataSource           The data source from which the chunk should be loaded.
   * @param dataSpec             The data spec that defines the data to be loaded.
   * @param trackFormat          The format of the track.
   * @param trackSelectionReason The reason for selecting this track.
   * @param trackSelectionData   The track selection data.
   * @param startTimeUs          The start time of the media contained by the chunk, in microseconds.
   * @param endTimeUs            The end time of the media contained by the chunk, in microseconds.
   * @param clippedStartTimeUs   The time in the chunk from which output will begin, or {@link
   *                             C#TIME_UNSET} to output from the start of the chunk.
   * @param clippedEndTimeUs     The time in the chunk from which output will end, or {@link
   *                             C#TIME_UNSET} to output to the end of the chunk.
   * @param chunkIndex           The index of the chunk, or {@link C#INDEX_UNSET} if it is not known.
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
   * Initializes the chunk for loading, setting the {@link BaseMediaChunkOutput} that will receive
   * samples as they are loaded.
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
   * @return The index of the first sample in the track.
   */
  public final int getFirstSampleIndex(int trackIndex) {
    return firstSampleIndices[trackIndex];
  }

  /**
   * Returns the output most recently passed to {@link #init(BaseMediaChunkOutput)}.
   *
   * @return The output.
   */
  protected final BaseMediaChunkOutput getOutput() {
    return output;
  }

  /**
   * Returns the clipped start time of the chunk.
   *
   * @return The clipped start time, in microseconds.
   */
  public final long getClippedStartTimeUs() {
    return clippedStartTimeUs;
  }

  /**
   * Returns the clipped end time of the chunk.
   *
   * @return The clipped end time, in microseconds.
   */
  public final long getClippedEndTimeUs() {
    return clippedEndTimeUs;
  }
}