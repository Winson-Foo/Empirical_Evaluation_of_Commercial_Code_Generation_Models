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

  private final DataSource upstream;
  private final DataSink dataSink;

  private boolean dataSinkNeedsClosing;
  private long bytesRemaining;

  /**
   * @param upstream The upstream {@link DataSource}.
   * @param dataSink The {@link DataSink} into which data is written.
   */
  public TeeDataSource(DataSource upstream, DataSink dataSink) {
    this.upstream = Assertions.checkNotNull(upstream);
    this.dataSink = Assertions.checkNotNull(dataSink);
  }

  /** Adds a transfer listener to the upstream data source. */
  @Override
  public void addTransferListener(TransferListener transferListener) {
    upstream.addTransferListener(transferListener);
  }

  /** Opens the upstream and data sink sources. */
  @Override
  public long open(DataSpec dataSpec) throws IOException {
    bytesRemaining = upstream.open(dataSpec);
    if (bytesRemaining == 0) {
      return 0;
    }
    if (dataSpec.length == C.LENGTH_UNSET && bytesRemaining != C.LENGTH_UNSET) {
      // Reconstruct dataSpec in order to provide the resolved length to the sink.
      dataSpec = dataSpec.subrange(0, bytesRemaining);
    }
    openDataSink(dataSpec);
    return bytesRemaining;
  }

  /** Reads from the upstream source and writes to the data sink. */
  @Override
  public int read(byte[] buffer, int offset, int max) throws IOException {
    if (bytesRemaining == 0) {
      return C.RESULT_END_OF_INPUT;
    }
    int bytesRead = upstream.read(buffer, offset, max);
    if (bytesRead > 0) {
      // TODO: Consider continuing even if writes to the sink fail.
      dataSink.write(buffer, offset, bytesRead);
      if (bytesRemaining != C.LENGTH_UNSET) {
        bytesRemaining -= bytesRead;
      }
    }
    return bytesRead;
  }

  /** Gets the URI of the upstream source. */
  @Override
  public @Nullable Uri getUri() {
    return upstream.getUri();
  }

  /** Gets the response headers of the upstream source. */
  @Override
  public Map<String, List<String>> getResponseHeaders() {
    return upstream.getResponseHeaders();
  }

  /** Closes the upstream and data sink sources. */
  @Override
  public void close() throws IOException {
    try {
      upstream.close();
    } finally {
      closeDataSink();
    }
  }

  /** Opens the data sink source. */
  private void openDataSink(DataSpec dataSpec) throws IOException {
    dataSinkNeedsClosing = true;
    dataSink.open(dataSpec);
  }

  /** Closes the data sink source. */
  private void closeDataSink() throws IOException {
    if (dataSinkNeedsClosing) {
      dataSinkNeedsClosing = false;
      dataSink.close();
    }
  }
} 