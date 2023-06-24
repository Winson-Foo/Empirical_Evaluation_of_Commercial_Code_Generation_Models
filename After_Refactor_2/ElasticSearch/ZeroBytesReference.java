package org.elasticsearch.common.bytes;

import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.BytesRefIterator;
import org.elasticsearch.common.util.PageCacheRecycler;

/**
 * A {@link BytesReference} of the given length which contains all zeroes.
 */
public class ZeroBytesReference extends AbstractBytesReference {

    private static final byte ZERO_BYTE = 0;
    private static final int MAX_BYTE_PAGE_SIZE = PageCacheRecycler.BYTE_PAGE_SIZE;

    public ZeroBytesReference(int length) {
        super(length);
        assert length >= 0 : "Length must be non-negative.";
    }

    /**
     * Finds the position of the first occurrence of the given byte in this
     * zero reference, starting at the specified index. Returns -1 if the byte
     * is not found.
     */
    @Override
    public int findBytePosition(byte byteToFind, int startIndex) {
        assert startIndex >= 0 && startIndex <= length : "startIndex must be in range [0, length].";
        if (byteToFind == ZERO_BYTE && startIndex < length) {
            return startIndex;
        } else {
            return -1;
        }
    }

    /**
     * Returns the byte at the given index in this zero reference.
     */
    @Override
    public byte get(int index) {
        assert index >= 0 && index < length : "Index must be in range [0, length).";
        return ZERO_BYTE;
    }

    /**
     * Returns a new zero reference that is a slice of this reference, starting
     * at the specified index and extending for the given length.
     */
    @Override
    public BytesReference slice(int fromIndex, int sliceLength) {
        assert fromIndex + sliceLength <= length : "Slice range must be within the bounds of the original reference.";
        return new ZeroBytesReference(sliceLength);
    }

    /**
     * Returns the amount of memory used by this zero reference (always 0).
     */
    @Override
    public long ramBytesUsed() {
        return 0;
    }

    /**
     * Returns a {@link BytesRef} representation of this zero reference.
     */
    @Override
    public BytesRef toBytesRef() {
        return new BytesRef(new byte[length], 0, length);
    }

    /**
     * Returns an iterator over the bytes in this zero reference. If the
     * reference length exceeds the maximum byte page size, the iterator will
     * return byte page-sized {@link BytesRef} objects.
     */
    @Override
    public BytesRefIterator iterator() {
        if (length <= MAX_BYTE_PAGE_SIZE) {
            return super.iterator();
        }
        final byte[] buffer = new byte[MAX_BYTE_PAGE_SIZE];
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