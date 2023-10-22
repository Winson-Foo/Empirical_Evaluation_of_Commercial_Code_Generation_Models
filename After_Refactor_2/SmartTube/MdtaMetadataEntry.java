package com.google.android.exoplayer2.extractor.mp4;

import androidx.annotation.Nullable;

import java.util.Arrays;

/**
 * Stores extensible metadata with handler type 'mdta'. See also the QuickTime File Format
 * Specification.
 */
class MdtaMetadataEntry implements Metadata.Entry {

    /** The metadata key name. */
    final String key;
    /** The payload. The interpretation of the value depends on {@link #typeIndicator}. */
    final byte[] value;
    /** The four byte locale indicator. */
    final int localeIndicator;
    /** The four byte type indicator. */
    final int typeIndicator;

    /**
     * Constructs a new metadata entry for the specified metadata key/value.
     *
     * @param key The metadata key name.
     * @param value The payload. The interpretation of the value depends on {@link #typeIndicator}.
     * @param localeIndicator The four byte locale indicator.
     * @param typeIndicator The four byte type indicator.
     */
    MdtaMetadataEntry(String key, byte[] value, int localeIndicator, int typeIndicator) {
        this.key = key;
        this.value = value;
        this.localeIndicator = localeIndicator;
        this.typeIndicator = typeIndicator;
    }

    @Override
    public boolean equals(@Nullable Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null || getClass() != obj.getClass()) {
            return false;
        }
        MdtaMetadataEntry other = (MdtaMetadataEntry) obj;
        return key.equals(other.key)
                && Arrays.equals(value, other.value)
                && localeIndicator == other.localeIndicator
                && typeIndicator == other.typeIndicator;
    }

    @Override
    public int hashCode() {
        int result = 17;
        result = 31 * result + key.hashCode();
        result = 31 * result + Arrays.hashCode(value);
        result = 31 * result + localeIndicator;
        result = 31 * result + typeIndicator;
        return result;
    }

    @Override
    public String toString() {
        return "mdta: key=" + key;
    }
}

