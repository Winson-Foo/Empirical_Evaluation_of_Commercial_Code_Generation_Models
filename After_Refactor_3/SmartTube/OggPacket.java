/**
 * OGG packet class.
 */
/* package */ final class OggPacket {

  private final OggPageHeader pageHeader = new OggPageHeader();
  private final ParsableByteArray packetData = new ParsableByteArray(new byte[OggPageHeader.MAX_PAGE_PAYLOAD], 0);
  private int currentSegmentIndex = C.INDEX_UNSET;
  private int segmentCount;
  private boolean populated;

  /**
   * Resets this reader.
   */
  public void reset() {
    pageHeader.reset();
    packetData.reset();
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
  public boolean readPacket(ExtractorInput input) throws IOException, InterruptedException {
    Assertions.checkState(input != null);

    if (populated) {
      populated = false;
      packetData.reset();
    }

    if (currentSegmentIndex < 0) {
      // Start of a new packet.
      if (!readPageHeader(input)) {
        // End of input reached without a valid page header.
        return false;
      }

      // Skip the page header.
      input.skipFully(pageHeader.headerSize);

      currentSegmentIndex = 0;
    }

    populatePacketData(input);
    return true;
  }

  /**
   * Reads the page header and updates the {@code pageHeader} field.
   *
   * @param input The {@link ExtractorInput} to read data from.
   * @return {@code true} if a valid page header was read, {@code false} otherwise.
   * @throws IOException If reading from the input fails.
   * @throws InterruptedException If the thread is interrupted.
   */
  private boolean readPageHeader(ExtractorInput input) throws IOException, InterruptedException {
    return pageHeader.populate(input, true);
  }

  /**
   * Reads the page segments for the current packet and updates the {@code packetData} field.
   *
   * @param input The {@link ExtractorInput} to read data from.
   * @throws IOException If reading from the input fails.
   * @throws InterruptedException If the thread is interrupted.
   */
  private void populatePacketData(ExtractorInput input) throws IOException, InterruptedException {
    while (!populated) {
      int packetSize = calculatePacketSize(currentSegmentIndex);

      // Resize the packet data buffer if necessary.
      if (packetData.capacity() < packetData.limit() + packetSize) {
        packetData.grow(packetData.limit() + packetSize - packetData.capacity());
      }

      // Read the next segment of the packet.
      input.readFully(packetData.data, packetData.limit(), packetSize);
      packetData.setLimit(packetData.limit() + packetSize);

      // Check if this is the last segment of the packet.
      populated = pageHeader.laces[currentSegmentIndex + segmentCount - 1] != 255;

      // Move to the next segment.
      currentSegmentIndex += segmentCount;
      if (currentSegmentIndex == pageHeader.pageSegmentCount) {
        // End of packet reached.
        currentSegmentIndex = C.INDEX_UNSET;
        break;
      }
    }
  }

  /**
   * Calculates the size of the current packet starting from {@code startSegmentIndex}.
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
        // End of packet reached.
        break;
      }
    }

    return size;
  }

  /**
   * Returns the {@link OggPageHeader} of the last page read, or an empty header if the packet has
   * yet to be populated.
   *
   * <p>Note that the returned {@link OggPageHeader} is mutable and may be updated during subsequent
   * calls to {@link #readPacket(ExtractorInput)}.
   *
   * @return the {@code PageHeader} of the last page read or an empty header if the packet has yet
   *     to be populated.
   */
  public OggPageHeader getPageHeader() {
    return pageHeader;
  }

  /**
   * Returns a {@link ParsableByteArray} containing the packet's payload.
   */
  public ParsableByteArray getPayload() {
    return packetData;
  }

  /**
   * Trims the packet data array.
   */
  public void trimPayload() {
    if (packetData.data.length == OggPageHeader.MAX_PAGE_PAYLOAD) {
      return;
    }

    packetData.data = Arrays.copyOf(packetData.data, Math.max(OggPageHeader.MAX_PAGE_PAYLOAD, packetData.limit()));
  }

} 