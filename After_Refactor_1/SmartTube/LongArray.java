/**
 * An append-only, auto-growing {@code long[]}.
 */
public final class LongArray {

  private static final int RESIZE_FACTOR = 2;

  private int count;
  private long[] values;

  /**
   * Constructs a new LongArray with the specified initial capacity.
   *
   * @param initialCapacity The initial capacity of the array.
   * @throws IllegalArgumentException If the initial capacity is less than or equal to zero.
   */
  public LongArray(int initialCapacity) {
    if (initialCapacity <= 0) {
      throw new IllegalArgumentException("Invalid initial capacity: " + initialCapacity);
    }
    values = new long[initialCapacity];
  }

  /**
   * Constructs a new LongArray with the specified values.
   *
   * @param values The values to initialize the array with.
   * @throws NullPointerException If values is null.
   */
  public LongArray(long[] values) {
    if (values == null) {
      throw new NullPointerException("Values cannot be null");
    }
    count = values.length;
    this.values = Arrays.copyOf(values, count);
  }

  /**
   * Appends a value to the end of the array.
   *
   * @param value The value to append.
   */
  public void add(long value) {
    resizeIfNeeded();
    values[count++] = value;
  }

  /**
   * Returns the value at the specified index.
   *
   * @param index The index of the value to retrieve.
   * @return The value at the specified index.
   * @throws IndexOutOfBoundsException If the index is less than zero, or greater than or equal to
   *         {@link #size()}.
   */
  public long get(int index) {
    if (index < 0 || index >= count) {
      throw new IndexOutOfBoundsException("Invalid index " + index + ", size is " + count);
    }
    return values[index];
  }

  /**
   * Returns the number of elements in the array.
   *
   * @return The number of elements in the array.
   */
  public int size() {
    return count;
  }

  /**
   * Returns a copy of the array as a new {@code long[]}.
   *
   * @return A new {@code long[]} containing the same elements as this array.
   */
  public long[] toArray() {
    return Arrays.copyOf(values, count);
  }

  /**
   * Resizes the array if it is full.
   */
  private void resizeIfNeeded() {
    if (count == values.length) {
      int newCapacity = values.length * RESIZE_FACTOR;
      long[] newValues = new long[newCapacity];
      System.arraycopy(values, 0, newValues, 0, count);
      values = newValues;
    }
  }

}