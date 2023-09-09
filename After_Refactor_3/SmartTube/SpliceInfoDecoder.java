package com.google.android.exoplayer2.metadata.scte35;

import com.google.android.exoplayer2.metadata.Metadata;
import com.google.android.exoplayer2.metadata.MetadataDecoder;
import com.google.android.exoplayer2.metadata.MetadataInputBuffer;
import com.google.android.exoplayer2.util.ParsableBitArray;
import com.google.android.exoplayer2.util.ParsableByteArray;
import com.google.android.exoplayer2.util.TimestampAdjuster;

import java.nio.ByteBuffer;

/**
 * Decodes splice info sections and produces splice commands.
 */
public final class SpliceInfoDecoder implements MetadataDecoder {

  private static final int TYPE_SPLICE_NULL = 0x00;
  private static final int TYPE_SPLICE_SCHEDULE = 0x04;
  private static final int TYPE_SPLICE_INSERT = 0x05;
  private static final int TYPE_TIME_SIGNAL = 0x06;
  private static final int TYPE_PRIVATE_COMMAND = 0xFF;

  private static final int BITS_TO_SKIP_BEFORE_SPLICE_COMMAND_LENGTH = 14;
  private static final int BITS_TO_SKIP_BEFORE_SPLICE_COMMAND_TYPE = 12;

  private final ParsableByteArray sectionData;
  private final ParsableBitArray sectionHeader;

  private TimestampAdjuster timestampAdjuster;

  public SpliceInfoDecoder() {
    sectionData = new ParsableByteArray();
    sectionHeader = new ParsableBitArray();
  }

  @SuppressWarnings("ByteBufferBackingArray")
  @Override
  public Metadata decode(MetadataInputBuffer inputBuffer) {
    createTimestampAdjusterIfNeeded(inputBuffer);

    ByteBuffer buffer = inputBuffer.data;
    byte[] sectionDataBytes = buffer.array();
    int sectionDataSize = buffer.limit();

    sectionData.reset(sectionDataBytes, sectionDataSize);
    sectionHeader.reset(sectionDataBytes, sectionDataSize);

    skipIrrelevantBitsInHeader();
    long ptsAdjustment = readPtsAdjustment();
    skipIrrelevantBitsInHeader();
    int spliceCommandLength = readSpliceCommandLength();
    int spliceCommandType = readSpliceCommandType();

    SpliceCommand command = parseSpliceCommand(spliceCommandType, ptsAdjustment);

    return command == null ? new Metadata() : new Metadata(command);
  }

  private void createTimestampAdjusterIfNeeded(MetadataInputBuffer inputBuffer) {
    if (timestampAdjuster == null
            || inputBuffer.subsampleOffsetUs != timestampAdjuster.getTimestampOffsetUs()) {
      timestampAdjuster = new TimestampAdjuster(inputBuffer.timeUs);
      timestampAdjuster.adjustSampleTimestamp(inputBuffer.timeUs - inputBuffer.subsampleOffsetUs);
    }
  }

  private void skipIrrelevantBitsInHeader() {
    sectionHeader.skipBits(BITS_TO_SKIP_BEFORE_SPLICE_COMMAND_LENGTH);
  }

  private long readPtsAdjustment() {
    long ptsAdjustment = sectionHeader.readBits(1);
    ptsAdjustment = (ptsAdjustment << 32) | sectionHeader.readBits(32);
    return ptsAdjustment;
  }

  private int readSpliceCommandLength() {
    sectionHeader.skipBits(BITS_TO_SKIP_BEFORE_SPLICE_COMMAND_TYPE);
    return sectionHeader.readBits(12);
  }

  private int readSpliceCommandType() {
    return sectionHeader.readBits(8);
  }

  private SpliceCommand parseSpliceCommand(int spliceCommandType, long ptsAdjustment) {
    SpliceCommand command = null;
    switch (spliceCommandType) {
      case TYPE_SPLICE_NULL:
        command = new SpliceNullCommand();
        break;
      case TYPE_SPLICE_SCHEDULE:
        command = SpliceScheduleCommand.parseFromSection(sectionData);
        break;
      case TYPE_SPLICE_INSERT:
        command = SpliceInsertCommand.parseFromSection(sectionData, ptsAdjustment, timestampAdjuster);
        break;
      case TYPE_TIME_SIGNAL:
        command = TimeSignalCommand.parseFromSection(sectionData, ptsAdjustment, timestampAdjuster);
        break;
      case TYPE_PRIVATE_COMMAND:
        command = PrivateCommand.parseFromSection(sectionData, readSpliceCommandLength(), ptsAdjustment);
        break;
      default:
        // Do nothing.
        break;
    }
    return command;
  }
} 