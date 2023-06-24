package com.google.android.exoplayer2.extractor.ogg;

import com.google.android.exoplayer2.C;
import com.google.android.exoplayer2.extractor.ExtractorInput;
import com.google.android.exoplayer2.util.Assertions;
import com.google.android.exoplayer2.util.ParsableByteArray;
import java.io.IOException;
import java.util.Arrays;

/**
 * Class representing an OGG packet.
 */
/* package */ final class OggPacket {

  private final OggPageHeader pageHeader = new OggPageHeader();
  private final ParsableByteArray payloadArray = new ParsableByteArray(
      new byte[OggPageHeader.MAX_PAGE_PAYLOAD], 0);

  private int currentSegmentIndex = C.INDEX_UNSET;
  private int segmentCount;
  private boolean isPopulated;

  /**
   * Resets this instance.
   */
  public void reset() {
    helperReset();
  }

  /**
   * Reads the next packet of the OGG stream.
   *
   * @param extractorInput The input stream.
   * @return {@code true} if the read was successful, {@code false} otherwise.
   * @throws IOException If reading from the input fails.
   * @throws InterruptedException If the thread is interrupted.
   */
  public boolean populate(ExtractorInput extractorInput) throws IOException, InterruptedException {
    Assertions.checkState(extractorInput != null);

    if (isPopulated) {
      helperReset();
    }

    while (!isPopulated) {
      if (currentSegmentIndex < 0) {
        pageHeader.populate(extractorInput, true);

        int segmentIndex = 0;
        int bytesToSkip = pageHeader.headerSize;
        if ((pageHeader.type & 0x01) == 0x01 && payloadArray.limit() == 0) {
          bytesToSkip += calculatePacketSize(segmentIndex);
          segmentIndex += segmentCount;
        }

        extractorInput.skipFully(bytesToSkip);
        currentSegmentIndex = segmentIndex;
      }

      int size = calculatePacketSize(currentSegmentIndex);
      int segmentIndex = currentSegmentIndex + segmentCount;
      if (size > 0) {
        ensureCapacity(size);
        extractorInput.readFully(payloadArray.data, payloadArray.limit(), size);
        payloadArray.setLimit(payloadArray.limit() + size);
        isPopulated = pageHeader.laces[segmentIndex - 1] != 255;
      }

      currentSegmentIndex = segmentIndex == pageHeader.pageSegmentCount ? C.INDEX_UNSET : segmentIndex;
    }

    return true;
  }

  /**
   * Returns the page header of the packet.
   *
   * @return The page header of the packet.
   */
  public OggPageHeader getPageHeader() {
    return pageHeader;
  }

  /**
   * Returns a {@link ParsableByteArray} containing the packet's payload.
   *
   * @return The packet's payload.
   */
  public ParsableByteArray getPayload() {
    return payloadArray;
  }

  /**
   * Trims the payload data array.
   */
  public void trimPayload() {
    if (payloadArray.data.length == OggPageHeader.MAX_PAGE_PAYLOAD) {
      return;
    }

    int newLength = Math.max(OggPageHeader.MAX_PAGE_PAYLOAD, payloadArray.limit());
    payloadArray.data = Arrays.copyOf(payloadArray.data, newLength);
  }

  /**
   * Calculates the size of the packet starting from {@code startSegmentIndex}.
   *
   * @param startSegmentIndex The index of the first segment of the packet.
   * @return The size of the packet.
   */
  private int calculatePacketSize(int startSegmentIndex) {
    segmentCount = 0;
    int size = 0;

    while (startSegmentIndex + segmentCount < pageHeader.pageSegmentCount) {
      int segmentLength = pageHeader.laces[startSegmentIndex + segmentCount++];
      size += segmentLength;

      if (segmentLength != 255) {
        break;
      }
    }

    return size;
  }

  /**
   * Resets variables to default values.
   */
  private void helperReset() {
    pageHeader.reset();
    payloadArray.reset();
    currentSegmentIndex = C.INDEX_UNSET;
    isPopulated = false;
  }

  /**
   * Ensures that the data array has the required capacity.
   *
   * @param requiredCapacity The required capacity of the array.
   */
  private void ensureCapacity(int requiredCapacity) {
    if (payloadArray.capacity() < payloadArray.limit() + requiredCapacity) {
      payloadArray.data = Arrays.copyOf(payloadArray.data, payloadArray.limit() + requiredCapacity);
    }
  }

} 