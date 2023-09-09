public class LongArray {

  private static final double GROWTH_FACTOR = 1.8;

  private long[] values;
  private int size;

  public static LongArray createWithInitialCapacity(int initialCapacity) {
    return new LongArray(initialCapacity);
  }

  private LongArray(int initialCapacity) {
    values = new long[initialCapacity];
    size = 0;
  }

  public void add(long value) {
    growArrayIfNeeded();
    values[size++] = value;
  }

  public long get(int index) {
    try {
      if (index >= size) {
        throw new IllegalArgumentException(
            String.format("Index %d is out of bounds. Array size is %d.", index, size));
      }
      return values[index];
    } catch (IndexOutOfBoundsException e) {
      throw new IllegalArgumentException(
          String.format("Index %d is out of bounds. Array size is %d.", index, size), e);
    }
  }

  public void set(int index, long value) {
    try {
      if (index >= size) {
        throw new IllegalArgumentException(
            String.format("Index %d is out of bounds. Array size is %d.", index, size));
      }
      values[index] = value;
    } catch (IndexOutOfBoundsException e) {
      throw new IllegalArgumentException(
          String.format("Index %d is out of bounds. Array size is %d.", index, size), e);
    }
  }

  public int size() {
    return size;
  }

  public boolean isEmpty() {
    return size == 0;
  }

  public void dropTail(int n) {
    if (n > size) {
      throw new IllegalArgumentException(
          String.format("Trying to drop %d items from array of length %d.", n, size));
    }
    size -= n;
  }

  private void growArrayIfNeeded() {
    if (size == values.length) {
      int newSize = Math.max(size + 1, (int) (size * GROWTH_FACTOR));
      long[] newArray = new long[newSize];
      System.arraycopy(values, 0, newArray, 0, size);
      values = newArray;
    }
  }
}

