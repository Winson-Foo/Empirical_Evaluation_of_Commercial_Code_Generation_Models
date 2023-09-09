package com.google.android.exoplayer2.metadata.scte35;

import com.google.android.exoplayer2.metadata.Metadata;
import com.google.android.exoplayer2.metadata.MetadataDecoder;
import com.google.android.exoplayer2.metadata.MetadataInputBuffer;
import com.google.android.exoplayer2.util.ParsableBitArray;
import com.google.android.exoplayer2.util.ParsableByteArray;
import com.google.android.exoplayer2.util.TimestampAdjuster;
import java.nio.ByteBuffer;

public final class SpliceInfoDecoder implements MetadataDecoder {

  private static final int TYPE_SPLICE_NULL = 0x00;
  private static final int TYPE_SPLICE_SCHEDULE = 0x04;
  private static final int TYPE_SPLICE_INSERT = 0x05;
  private static final int TYPE_TIME_SIGNAL = 0x06;
  private static final int TYPE_PRIVATE_COMMAND = 0xFF;

  private final ParsableByteArray sectionData;
  private final ParsableBitArray sectionHeader;

  private TimestampAdjuster timestampAdjuster;

  public SpliceInfoDecoder() {
    sectionData = new ParsableByteArray();
    sectionHeader = new ParsableBitArray();
  }

  @Override
  public Metadata decode(MetadataInputBuffer inputBuffer) {
    initializeTimestampAdjuster(inputBuffer);

    ByteBuffer buffer = inputBuffer.data;
    byte[] data = buffer.array();
    int size = buffer.limit();
    sectionData.reset(data, size);
    sectionHeader.reset(data, size);

    long ptsAdjustment = readPtsAdjustment();
    int spliceCommandLength = readSpliceCommandLength();
    int spliceCommandType = readSpliceCommandType();
    SpliceCommand command = readSpliceCommand(spliceCommandType, ptsAdjustment);

    return command == null ? new Metadata() : new Metadata(command);
  }

  private void initializeTimestampAdjuster(MetadataInputBuffer inputBuffer) {
    if (timestampAdjuster == null
        || inputBuffer.subsampleOffsetUs != timestampAdjuster.getTimestampOffsetUs()) {
      timestampAdjuster = new TimestampAdjuster(inputBuffer.timeUs);
      timestampAdjuster.adjustSampleTimestamp(inputBuffer.timeUs - inputBuffer.subsampleOffsetUs);
    }
  }

  private long readPtsAdjustment() {
    sectionHeader.skipBits(39);
    long ptsAdjustment = sectionHeader.readBits(1);
    ptsAdjustment = (ptsAdjustment << 32) | sectionHeader.readBits(32);
    sectionHeader.skipBits(20);
    return ptsAdjustment;
  }

  private int readSpliceCommandLength() {
    return sectionHeader.readBits(12);
  }

  private int readSpliceCommandType() {
    return sectionHeader.readBits(8);
  }

  private SpliceCommand readSpliceCommand(int spliceCommandType, long ptsAdjustment) {
    sectionData.skipBytes(14);

    switch (spliceCommandType) {
      case TYPE_SPLICE_NULL:
        return new SpliceNullCommand();
      case TYPE_SPLICE_SCHEDULE:
        return SpliceScheduleCommand.parseFromSection(sectionData);
      case TYPE_SPLICE_INSERT:
        return SpliceInsertCommand.parseFromSection(sectionData, ptsAdjustment, timestampAdjuster);
      case TYPE_TIME_SIGNAL:
        return TimeSignalCommand.parseFromSection(sectionData, ptsAdjustment, timestampAdjuster);
      case TYPE_PRIVATE_COMMAND:
        return PrivateCommand.parseFromSection(sectionData, spliceCommandLength, ptsAdjustment);
      default:
        return null;
    }
  }

} 