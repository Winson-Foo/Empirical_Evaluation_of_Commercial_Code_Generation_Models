package com.google.android.exoplayer2.source.chunk

import com.google.android.exoplayer2.C
import com.google.android.exoplayer2.Format
import com.google.android.exoplayer2.upstream.DataSource
import com.google.android.exoplayer2.upstream.DataSpec

/**
 * A base implementation of [MediaChunk] that outputs to a [MediaChunkOutput].
 *
 * @param clippedStartTimeUs The time in the chunk from which output will begin, or [C.TIME_UNSET] to
 * output from the start of the chunk.
 * @param clippedEndTimeUs The time in the chunk from which output will end, or [C.TIME_UNSET] to output
 * to the end of the chunk.
 * @param chunkIndex The index of the chunk, or [C.INDEX_UNSET] if it is not known.
 */
abstract class BaseMediaChunk(
    dataSource: DataSource,
    dataSpec: DataSpec,
    trackFormat: Format,
    trackSelectionReason: Int,
    trackSelectionData: Any?,
    startTimeUs: Long,
    endTimeUs: Long,
    val clippedStartTimeUs: Long,
    val clippedEndTimeUs: Long,
    chunkIndex: Long
) : MediaChunk(
    dataSource,
    dataSpec,
    trackFormat,
    trackSelectionReason,
    trackSelectionData,
    startTimeUs,
    endTimeUs,
    chunkIndex
) {
    /** The output that will receive the loaded media samples. */
    private var output: MediaChunkOutput? = null

    /** The index of the first sample in each of the chunk track samples arrays. */
    private var firstSampleIndices: IntArray = IntArray(0)

    /**
     * Calls [MediaChunkOutputProvider] to get the [MediaChunkOutput] that will receive samples as they
     * are loaded.
     */
    override fun init(outputProvider: MediaChunkOutputProvider) {
        output = outputProvider.getMediaChunkOutput(trackFormat)
        firstSampleIndices = output!!.getWriteIndices()
    }

    /** Returns the output most recently passed to [init]. */
    protected fun getOutput(): MediaChunkOutput? {
        return output
    }

    /** Returns the index of the first sample in the specified track of the output that will originate from this chunk. */
    protected fun getFirstSampleIndex(trackIndex: Int): Int {
        return firstSampleIndices[trackIndex]
    }

    override fun release() {
        output = null
    }
} 