package com.google.android.exoplayer2.util;

import java.util.Arrays;

public final class LongArray {

  private static final int DEFAULT_INITIAL_CAPACITY = 32;

  private int size;
  private long[] array;

  public LongArray() {
    this(DEFAULT_INITIAL_CAPACITY);
  }

  public LongArray(int initialCapacity) {
    array = new long[initialCapacity];
  }

  public void add(long value) {
    if (size == array.length) {
      array = Arrays.copyOf(array, size * 2);
    }
    array[size++] = value;
  }

  public long get(int index) {
    if (index < 0 || index >= size) {
      throw new IndexOutOfBoundsException("Invalid index " + index + ", size is " + size);
    }
    return array[index];
  }

  public long[] toArray() {
    return Arrays.copyOf(array, size);
  }

  public int size() {
    return size;
  }
}

