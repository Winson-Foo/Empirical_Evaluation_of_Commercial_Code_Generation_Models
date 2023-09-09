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
        // Internal timestamps adjustment.
        maybeAdjustTimestampOffset(inputBuffer.timeUs, inputBuffer.subsampleOffsetUs);

        ByteBuffer buffer = inputBuffer.data;
        byte[] data = buffer.array();
        int size = buffer.limit();

        sectionData.reset(data, size);
        sectionHeader.reset(data, size);
        skipHeaderFields();

        long ptsAdjustment = readPtsAdjustment();
        int spliceCommandType = readSpliceCommandType();
        int spliceCommandLength = readSpliceCommandLength();

        SpliceCommand command = null;

        switch (spliceCommandType) {
            case TYPE_SPLICE_NULL:
                command = new SpliceNullCommand();
                break;
            case TYPE_SPLICE_SCHEDULE:
                command = SpliceScheduleCommand.parseFromSection(sectionData);
                break;
            case TYPE_SPLICE_INSERT:
                command = SpliceInsertCommand.parseFromSection(sectionData, ptsAdjustment,
                        timestampAdjuster);
                break;
            case TYPE_TIME_SIGNAL:
                command = TimeSignalCommand.parseFromSection(sectionData, ptsAdjustment, timestampAdjuster);
                break;
            case TYPE_PRIVATE_COMMAND:
                command = PrivateCommand.parseFromSection(sectionData, spliceCommandLength, ptsAdjustment);
                break;
            default:
                // Do nothing.
                break;
        }

        return command == null ? new Metadata() : new Metadata(command);
    }

    private void maybeAdjustTimestampOffset(long timeUs, long subsampleOffsetUs) {
        if (timestampAdjuster == null || subsampleOffsetUs != timestampAdjuster.getTimestampOffsetUs()) {
            timestampAdjuster = new TimestampAdjuster(timeUs);
            timestampAdjuster.adjustSampleTimestamp(timeUs - subsampleOffsetUs);
        }
    }

    private void skipHeaderFields() {
        sectionHeader.skipBits(8 + 1 + 1 + 2 + 12 + 8 + 1 + 6);
        sectionHeader.skipBits(8 + 12);
    }

    private long readPtsAdjustment() {
        long ptsAdjustment = sectionHeader.readBits(1);
        ptsAdjustment = (ptsAdjustment << 32) | sectionHeader.readBits(32);
        return ptsAdjustment;
    }

    private int readSpliceCommandType() {
        return sectionHeader.readBits(8);
    }

    private int readSpliceCommandLength() {
        return sectionHeader.readBits(12);
    }
} 