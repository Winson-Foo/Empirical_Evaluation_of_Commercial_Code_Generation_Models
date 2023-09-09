public class LongArray {
  private static final double GROWTH_FACTOR = 1.8;
  private static final int DEFAULT_INITIAL_CAPACITY = 16;

  private long[] array;
  private int size;

  public LongArray() {
    this(DEFAULT_INITIAL_CAPACITY);
  }

  public LongArray(int initialCapacity) {
    array = new long[initialCapacity];
    size = 0;
  }

  public void add(long value) {
    growIfNeeded();
    array[size++] = value;
  }

  public long get(int index) {
    if (!isValidIndex(index)) {
      throw new IndexOutOfBoundsException("Index out of range: " + index);
    }
    return array[index];
  }

  public void set(int index, long value) {
    if (!isValidIndex(index)) {
      throw new IndexOutOfBoundsException("Index out of range: " + index);
    }
    array[index] = value;
  }

  public int size() {
    return size;
  }

  public boolean isEmpty() {
    return size == 0;
  }

  public void dropTail(int n) {
    if (n > size) {
      throw new IndexOutOfBoundsException("Cannot drop " + n + " items from an array of size " + size);
    }
    size -= n;
  }

  private void growIfNeeded() {
    if (size == array.length) {
      int newSize = Math.max(size + 1, (int) (size * GROWTH_FACTOR));
      long[] newArray = new long[newSize];
      System.arraycopy(array, 0, newArray, 0, size);
      array = newArray;
    }
  }

  private boolean isValidIndex(int index) {
    return index >= 0 && index < size;
  }
} 