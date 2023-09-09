package com.google.android.exoplayer2.upstream;

import android.net.Uri;
import androidx.annotation.Nullable;
import com.google.android.exoplayer2.util.Assertions;
import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Tees data into a {@link DataSink} as the data is read.
 */
public final class TeeDataSource implements DataSource {

  private final DataSource dataSource;
  private final DataSink sink;

  private boolean sinkNeedsClosing;
  private long remainingBytesToRead;

  /**
   * @param dataSource The upstream {@link DataSource}.
   * @param sink The {@link DataSink} into which data is written.
   */
  public TeeDataSource(DataSource dataSource, DataSink sink) {
    this.dataSource = Assertions.checkNotNull(dataSource);
    this.sink = Assertions.checkNotNull(sink);
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public void addTransferListener(TransferListener transferListener) {
    dataSource.addTransferListener(transferListener);
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public long open(DataSpec dataSpec) throws TeeDataSourceException {
    try {
      remainingBytesToRead = dataSource.open(dataSpec);
      if (remainingBytesToRead == 0) {
        return 0;
      }
      if (dataSpec.length == C.LENGTH_UNSET && remainingBytesToRead != C.LENGTH_UNSET) {
        // Reconstruct dataSpec in order to provide the resolved length to the sink.
        dataSpec = dataSpec.subrange(0, remainingBytesToRead);
      }
      sinkNeedsClosing = true;
      sink.open(dataSpec);
      return remainingBytesToRead;
    } catch (IOException e) {
      throw new TeeDataSourceException(e);
    }
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public int read(byte[] buffer, int offset, int length) throws TeeDataSourceException {
    try {
      if (remainingBytesToRead == 0) {
        return C.RESULT_END_OF_INPUT;
      }
      int bytesRead = dataSource.read(buffer, offset, length);
      if (bytesRead > 0) {
        // TODO: Consider continuing even if writes to the sink fail.
        sink.write(buffer, offset, bytesRead);
        if (remainingBytesToRead != C.LENGTH_UNSET) {
          remainingBytesToRead -= bytesRead;
        }
      }
      return bytesRead;
    } catch (IOException e) {
      throw new TeeDataSourceException(e);
    }
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public @Nullable Uri getUri() {
    return dataSource.getUri();
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public Map<String, List<String>> getResponseHeaders() {
    return dataSource.getResponseHeaders();
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public void close() throws TeeDataSourceException {
    try {
      dataSource.close();
    } catch (IOException e) {
      throw new TeeDataSourceException(e);
    } finally {
      if (sinkNeedsClosing) {
        sinkNeedsClosing = false;
        try {
          sink.close();
        } catch (IOException e) {
          throw new TeeDataSourceException(e);
        }
      }
    }
  }

  /**
   * Custom exception to wrap IO exceptions thrown while reading from or
   * writing to the underlying data sources/sinks.
   */
  public static class TeeDataSourceException extends IOException {
    public TeeDataSourceException(IOException cause) {
      super(cause);
    }
  }
} 