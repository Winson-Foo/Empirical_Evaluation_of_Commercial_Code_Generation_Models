// MetadataEntry.java
package com.google.android.exoplayer2.extractor.mp4;

import android.os.Parcel;
import android.os.Parcelable;

import androidx.annotation.Nullable;

import com.google.android.exoplayer2.metadata.Metadata;
import com.google.android.exoplayer2.util.Util;

import java.util.Arrays;

/** Stores metadata with a specific key and value. */
public final class MetadataEntry implements Metadata.Entry {

  public static final int LOCALE_INDICATOR_UNSPECIFIED = 0x00000000;

  /** The metadata key. */
  public final MetadataKey key;

  /** The metadata value. */
  public final byte[] value;

  /** The metadata locale indicator. */
  public final LocaleIndicator localeIndicator;

  /** The metadata type indicator. */
  public final TypeIndicator typeIndicator;

  /**
   * Creates a new metadata entry.
   *
   * @param key The metadata key.
   * @param value The metadata value.
   * @param localeIndicator The metadata locale indicator.
   * @param typeIndicator The metadata type indicator.
   */
  public MetadataEntry(
      MetadataKey key, byte[] value, LocaleIndicator localeIndicator, TypeIndicator typeIndicator) {
    this.key = key;
    this.value = value;
    this.localeIndicator = localeIndicator;
    this.typeIndicator = typeIndicator;
  }

  private MetadataEntry(Parcel in) {
    key = Util.castNonNull(MetadataKey.valueOf(in.readString()));
    value = new byte[in.readInt()];
    in.readByteArray(value);
    localeIndicator = Util.castNonNull(LocaleIndicator.valueOf(in.readInt()));
    typeIndicator = Util.castNonNull(TypeIndicator.valueOf(in.readInt()));
  }

  @Override
  public boolean equals(@Nullable Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null || getClass() != obj.getClass()) {
      return false;
    }
    MetadataEntry other = (MetadataEntry) obj;
    return key == other.key
        && Arrays.equals(value, other.value)
        && localeIndicator == other.localeIndicator
        && typeIndicator == other.typeIndicator;
  }

  @Override
  public int hashCode() {
    int result = 17;
    result = 31 * result + key.hashCode();
    result = 31 * result + Arrays.hashCode(value);
    result = 31 * result + localeIndicator.hashCode();
    result = 31 * result + typeIndicator.hashCode();
    return result;
  }

  @Override
  public String toString() {
    return "MetadataEntry: key=" + key.id();
  }

  @Override
  public void writeToParcel(Parcel dest, int flags) {
    dest.writeString(key.name());
    dest.writeInt(value.length);
    dest.writeByteArray(value);
    dest.writeInt(localeIndicator.value);
    dest.writeInt(typeIndicator.value);
  }

  @Override
  public int describeContents() {
    return 0;
  }

  /** An enum for all possible values of a metadata key. */
  public enum MetadataKey {
    Mdta,
    // ... Other metadata keys ...
  }

  /** An enum for all possible values of a metadata's locale indicator bytes. */
  public enum LocaleIndicator {
    Unspecified(MetadataEntry.LOCALE_INDICATOR_UNSPECIFIED);

    final int value;

    LocaleIndicator(int value) {
      this.value = value;
    }

    public static LocaleIndicator valueOf(int value) {
      for (LocaleIndicator indicator : values()) {
        if (indicator.value == value) {
          return indicator;
        }
      }
      return Unspecified;
    }
  }

  /** An enum for all possible values of a metadata's type indicator bytes. */
  public enum TypeIndicator {
    // ... Possible type indicators ...
  }
}

// MetadataEntryCreator.java
package com.google.android.exoplayer2.extractor.mp4;

import android.os.Parcel;
import android.os.Parcelable;

public final class MetadataEntryCreator implements Parcelable.Creator<MetadataEntry> {
  @Override
  public MetadataEntry createFromParcel(Parcel in) {
    return new MetadataEntry(in);
  }

  @Override
  public MetadataEntry[] newArray(int size) {
    return new MetadataEntry[size];
  }
} 