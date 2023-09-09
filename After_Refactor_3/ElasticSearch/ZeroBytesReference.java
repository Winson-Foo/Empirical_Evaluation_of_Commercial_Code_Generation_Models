package org.elasticsearch.common.bytes;

import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.BytesRefIterator;
import org.elasticsearch.common.util.PageCacheRecycler;

/**
 * A {@link BytesReference} of the given length which contains all zeroes.
 */
public class ZeroBytesReference extends AbstractBytesReference {

    private static final byte ZERO = 0;

    public ZeroBytesReference(int length) {
        super(length);
        assert length >= 0 : "length should be non-negative";
    }

    /**
     * Returns the index of the first occurrence of the specified byte in this byte sequence,
     * starting the search at the specified index.
     *
     * @param target the byte to search for.
     * @param from the index to start the search from.
     * @return the index of the first occurrence of the byte in the byte sequence,
     *         or -1 if the byte was not found.
     */
    @Override
    public int indexOf(byte target, int from) {
        assert 0 <= from && from <= length : "from (" + from + ") should be between 0 and " + length;

        if (target == ZERO && from < length) {
            return from;
        } else {
            return -1;
        }
    }

    /**
     * Returns the byte at the specified index.
     *
     * @param index the index of the byte.
     * @return the byte at the specified index.
     */
    @Override
    public byte get(int index) {
        assert 0 <= index && index < length : "index (" + index + ") should be between 0 and " + (length - 1);
        return ZERO;
    }

    /**
     * Returns a new byte sequence that is a subsequence of this byte sequence.
     *
     * @param from the start index (inclusive).
     * @param length the length of the subsequence.
     * @return the subsequence.
     */
    @Override
    public BytesReference slice(int from, int length) {
        assert from >= 0 && (from + length) <= this.length : "Invalid parameters for slice(): from = " + from + ", length = " + length;
        return new ZeroBytesReference(length);
    }

    /**
     * Returns an estimate of the memory usage of this byte sequence.
     *
     * @return the memory usage in bytes.
     */
    @Override
    public long ramBytesUsed() {
        return 0;
    }

    /**
     * Returns a {@link BytesRef} that represents this byte sequence.
     *
     * @return a new {@link BytesRef} instance.
     */
    @Override
    public BytesRef toBytesRef() {
        return new BytesRef(new byte[length], 0, length);
    }

    /**
     * Returns an iterator over the bytes in this byte sequence.
     * <p>
     * If the length of this byte sequence is greater than {@link PageCacheRecycler#BYTE_PAGE_SIZE},
     * the bytes will be returned in chunks of size {@link PageCacheRecycler#BYTE_PAGE_SIZE}.
     * </p>
     *
     * @return an iterator over the bytes in this byte sequence.
     */
    @Override
    public BytesRefIterator iterator() {
        if (length <= PageCacheRecycler.BYTE_PAGE_SIZE) {
            return super.iterator();
        }

        final byte[] buffer = new byte[PageCacheRecycler.BYTE_PAGE_SIZE];

        return new BytesRefIterator() {

            int remaining = length;

            @Override
            public BytesRef next() {
                if (remaining > 0) {
                    final int nextLength = Math.min(remaining, buffer.length);
                    remaining -= nextLength;
                    return new BytesRef(buffer, 0, nextLength);
                } else {
                    return null;
                }
            }
        };
    }
}