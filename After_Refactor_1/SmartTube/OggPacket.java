package com.google.android.exoplayer2.extractor.ogg;

import com.google.android.exoplayer2.C;
import com.google.android.exoplayer2.extractor.ExtractorInput;
import com.google.android.exoplayer2.util.Assertions;
import com.google.android.exoplayer2.util.ParsableByteArray;

import java.io.IOException;
import java.util.Arrays;

/** OGG packet class. */
public final class OggPacket {

  private final OggPageHeader pageHeader = new OggPageHeader();
  private final ParsableByteArray packetArray =
      new ParsableByteArray(new byte[OggPageHeader.MAX_PAGE_PAYLOAD], 0);

  private int currentSegmentIndex = C.INDEX_UNSET;
  private int segmentCount;
  private boolean populated;

  /** Resets this reader. */
  public void reset() {
    pageHeader.reset();
    packetArray.reset();
    currentSegmentIndex = C.INDEX_UNSET;
    populated = false;
  }

  /**
   * Reads the next packet of the ogg stream. In case of an {@code IOException} the caller must make
   * sure to pass the same instance of {@code ParsableByteArray} to this method again so this reader
   * can resume properly from an error while reading a continued packet spanned across multiple
   * pages.
   *
   * @param input The {@link ExtractorInput} to read data from.
   * @return {@code true} if the read was successful. The read fails if the end of the input is
   *     encountered without reading data.
   * @throws IOException If reading from the input fails.
   * @throws InterruptedException If the thread is interrupted.
   */
  public boolean populate(ExtractorInput input) throws IOException, InterruptedException {
    Assertions.checkState(input != null);

    if (populated) {
      populated = false;
      packetArray.reset();
    }

    while (!populated) {
      if (currentSegmentIndex < 0) {
        // We're at the start of a page.
        if (!pageHeader.populate(input, true)) {
          return false;
        }
        int segmentIndex = 0;
        int bytesToSkip = pageHeader.headerSize;
        if ((pageHeader.type & 0x01) == 0x01 && packetArray.limit() == 0) {
          // After seeking, the first packet may be the remainder
          // part of a continued packet which has to be discarded.
          bytesToSkip += calculatePacketSize(segmentIndex);
          segmentIndex += segmentCount;
        }
        input.skipFully(bytesToSkip);
        currentSegmentIndex = segmentIndex;
      }

      int size = calculatePacketSize(currentSegmentIndex);
      int segmentIndex = currentSegmentIndex + segmentCount;
      if (size > 0) {
        if (packetArray.capacity() < packetArray.limit() + size) {
          packetArray.ensureCapacity(packetArray.limit() + size);
        }
        input.readFully(packetArray.data, packetArray.limit(), size);
        packetArray.setLimit(packetArray.limit() + size);
        populated = pageHeader.laces[segmentIndex - 1] != 255;
      }
      // Advance now since we are sure reading didn't throw an exception.
      currentSegmentIndex = segmentIndex == pageHeader.pageSegmentCount ? C.INDEX_UNSET : segmentIndex;
    }
    return true;
  }

  /** An OGG Packet may span multiple pages. */
  public OggPageHeader getPageHeader() {
    return pageHeader;
  }

  /** Returns a {@link ParsableByteArray} containing the packet's payload. */
  public ParsableByteArray getPayload() {
    return packetArray;
  }

  /** Trims the packet data array. */
  public void trimPayload() {
    if (packetArray.limit() == packetArray.capacity()) {
      return;
    }
    packetArray.resize(packetArray.limit());
  }

  /**
   * Calculates the size of the packet starting from {@code startSegmentIndex}.
   *
   * @param startSegmentIndex the index of the first segment of the packet.
   * @return Size of the packet.
   */
  private int calculatePacketSize(int startSegmentIndex) {
    segmentCount = 0;
    int size = 0;
    while (startSegmentIndex + segmentCount < pageHeader.pageSegmentCount) {
      int segmentLength = pageHeader.laces[startSegmentIndex + segmentCount++];
      size += segmentLength;
      if (segmentLength != 255) {
        // packets end at first lace < 255
        break;
      }
    }
    return size;
  }
}