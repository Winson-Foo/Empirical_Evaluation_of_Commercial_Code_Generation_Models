package org.elasticsearch.common.bytes;

import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.BytesRefIterator;
import org.elasticsearch.common.util.PageCacheRecycler;

public class AllZeroBytesReference extends AbstractBytesReference {

    private static final byte ZERO_BYTE = 0;

    public AllZeroBytesReference(int length) {
        super(length);
        assert length >= 0 : length;
    }

    @Override
    public int indexOf(byte marker, int from) {
        assert from >= 0 && from <= length : from + " vs " + length;
        if (marker == ZERO_BYTE && from < length) {
            return from;
        }
        return -1;
    }

    @Override
    public byte get(int index) {
        assert index >= 0 && index < length : index + " vs " + length;
        return ZERO_BYTE;
    }

    @Override
    public BytesReference slice(int from, int length) {
        assert from + length <= this.length : from + " and " + length + " vs " + this.length;
        return new AllZeroBytesReference(length);
    }

    @Override
    public long ramBytesUsed() {
        return 0;
    }

    @Override
    public BytesRef toBytesRef() {
        return new BytesRef(new byte[length], 0, length);
    }

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
                }
                return null;
            }
        };
    }
}

