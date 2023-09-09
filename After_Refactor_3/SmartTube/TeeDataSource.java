package com.google.android.exoplayer2.upstream;

import android.net.Uri;

import androidx.annotation.Nullable;

import com.google.android.exoplayer2.C;
import com.google.android.exoplayer2.util.Assertions;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Tees data into a {@link DataSink} as the data is read.
 */
public final class TeeDataSource implements DataSource {

    private final DataSource mUpstream;
    private final DataSink mDataSink;

    private boolean mDataSinkNeedsClosing;
    private long mBytesRemaining;

    /**
     * @param upstream The upstream {@link DataSource}.
     * @param dataSink The {@link DataSink} into which data is written.
     */
    public TeeDataSource(final DataSource upstream, final DataSink dataSink) {
        mUpstream = Assertions.checkNotNull(upstream);
        mDataSink = Assertions.checkNotNull(dataSink);
    }

    @Override
    public void addTransferListener(final TransferListener transferListener) {
        mUpstream.addTransferListener(transferListener);
    }

    @Override
    public long open(final DataSpec dataSpec) throws IOException {
        mBytesRemaining = mUpstream.open(dataSpec);
        if (mBytesRemaining == 0) {
            return 0;
        }
        if (dataSpec.length == C.LENGTH_UNSET && mBytesRemaining != C.LENGTH_UNSET) {
            // Reconstruct dataSpec in order to provide the resolved length to the sink.
            dataSpec = dataSpec.subrange(0, mBytesRemaining);
        }
        mDataSinkNeedsClosing = true;
        mDataSink.open(dataSpec);
        return mBytesRemaining;
    }

    @Override
    public int read(final byte[] buffer, final int offset, final int max) throws IOException {
        if (mBytesRemaining == 0) {
            return C.RESULT_END_OF_INPUT;
        }
        int bytesRead = mUpstream.read(buffer, offset, max);
        if (bytesRead > 0) {
            // TODO: Consider continuing even if writes to the sink fail.
            mDataSink.write(buffer, offset, bytesRead);
            if (mBytesRemaining != C.LENGTH_UNSET) {
                mBytesRemaining -= bytesRead;
            }
        }
        return bytesRead;
    }

    @Override
    public @Nullable Uri getUri() {
        return mUpstream.getUri();
    }

    @Override
    public Map<String, List<String>> getResponseHeaders() {
        return mUpstream.getResponseHeaders();
    }

    @Override
    public void close() throws IOException {
        try (final DataSink ignored = mDataSinkNeedsClosing ? mDataSink : null) {
            mUpstream.close();
        }
    }

} 