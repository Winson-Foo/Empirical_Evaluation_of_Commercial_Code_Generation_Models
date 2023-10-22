package org.elasticsearch.common.geo;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Comparator;
import java.util.List;

import org.apache.lucene.util.BitUtil;
import org.elasticsearch.common.io.stream.BytesStreamOutput;
import org.elasticsearch.geometry.Rectangle;
import org.elasticsearch.search.aggregations.bucket.geogrid.GeoTileUtils;

public class SimpleFeatureFactory {

    private static final byte[] EMPTY = new byte[0];
    private static final int MOVETO = 1;
    private static final int LINETO = 2;
    private static final int CLOSEPATH = 7;

    private final int extent;
    private final double pointXScale;
    private final double pointYScale;
    private final double pointXTranslate;
    private final double pointYTranslate;

    public SimpleFeatureFactory(int zoom, int x, int y, int extent) {
        this.extent = extent;
        Rectangle rectangle = SphericalMercatorUtils.recToSphericalMercator(GeoTileUtils.toBoundingBox(x, y, zoom));
        pointXScale = (double) extent / (rectangle.getMaxLon() - rectangle.getMinLon());
        pointYScale = (double) -extent / (rectangle.getMaxLat() - rectangle.getMinLat());
        pointXTranslate = -pointXScale * rectangle.getMinX();
        pointYTranslate = -pointYScale * rectangle.getMinY();
    }

    public byte[] point(double lon, double lat) {
        int posLon = lonToPos(lon);
        int posLat = latToPos(lat);

        if (isOutboundingExtent(posLon) || isOutboundingExtent(posLat)) {
            return EMPTY;
        }

        try {
            return writePointCommands(posLon, posLat);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public byte[] points(List<GeoPoint> points) {
        points.sort(Comparator.comparingDouble(GeoPoint::getLon).thenComparingDouble(GeoPoint::getLat));

        int[] commands = new int[2 * points.size() + 1];
        int pos = 1;
        int currLon = 0, currLat = 0;
        int numPoints = 0;

        for (GeoPoint point : points) {
            int posLon = lonToPos(point.getLon());
            int posLat = latToPos(point.getLat());

            if (isOutboundingExtent(posLon) || isOutboundingExtent(posLat)) {
                continue;
            }

            if (numPoints == 0 || posLon != currLon || posLat != currLat) {
                commands[pos++] = BitUtil.zigZagEncode(posLon - currLon);
                commands[pos++] = BitUtil.zigZagEncode(posLat - currLat);
                currLon = posLon;
                currLat = posLat;
                numPoints++;
            }
        }

        if (numPoints == 0) {
            return EMPTY;
        }

        commands[0] = encodeCommand(MOVETO, numPoints);

        try {
            return writeCommands(commands, 1, pos);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public byte[] box(double minLon, double maxLon, double minLat, double maxLat) {
        int minX = Math.max(0, lonToPos(minLon));
        int minY = Math.min(extent, latToPos(minLat));
        int maxX = Math.min(extent, lonToPos(maxLon));
        int maxY = Math.max(0, latToPos(maxLat));

        if (isOutboundingExtent(minX) || isOutboundingExtent(minY) || isOutboundingExtent(maxX) || isOutboundingExtent(maxY)) {
            return EMPTY;
        }

        try {
            if (minX == maxX && minY == maxY) {
                return writePointCommands(minX, minY);
            } else if (minX == maxX) {
                return writeLineCommands(minX, minY, maxX, maxY);
            } else if (minY == maxY) {
                return writeLineCommands(minX, minY, maxX, minY);
            } else {
                return writeBoxCommands(minX, maxX, minY, maxY);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private boolean isOutboundingExtent(int pos) {
        return pos > extent || pos < 0;
    }

    private int lonToPos(double lon) {
        return (int) Math.round(pointXScale * SphericalMercatorUtils.lonToSphericalMercator(lon) + pointXTranslate);
    }

    private int latToPos(double lat) {
        return (int) Math.round(pointYScale * SphericalMercatorUtils.latToSphericalMercator(lat) + pointYTranslate) + extent;
    }

    private static int encodeCommand(int id, int length) {
        return (id & 0x7) | (length << 3);
    }

    private static byte[] writeCommands(int[] commands, int type, int length) throws IOException {
        try (BytesStreamOutput output = new BytesStreamOutput()) {
            for (int i = 0; i < length; i++) {
                output.writeVInt(commands[i]);
            }

            int dataSize = output.size();
            output.reset();
            output.writeVInt(24);
            output.writeVInt(type);
            output.writeVInt(34);
            output.writeVInt(dataSize);

            for (int i = 0; i < length; i++) {
                output.writeVInt(commands[i]);
            }

            return output.copyBytes().array();
        }
    }

    private static byte[] writePointCommands(int x, int y) throws IOException {
        int[] commands = new int[]{encodeCommand(MOVETO, 1), BitUtil.zigZagEncode(x), BitUtil.zigZagEncode(y)};
        return writeCommands(commands, 1, 3);
    }

    private static byte[] writeLineCommands(int x1, int y1, int x2, int y2) throws IOException {
        int[] commands = new int[]{encodeCommand(MOVETO, 1), BitUtil.zigZagEncode(x1), BitUtil.zigZagEncode(y1),
            encodeCommand(LINETO, 1), BitUtil.zigZagEncode(x2 - x1), BitUtil.zigZagEncode(y2 - y1)};
        return writeCommands(commands, 2, 6);
    }

    private static byte[] writeBoxCommands(int minX, int maxX, int minY, int maxY) throws IOException {
        int[] commands = new int[]{encodeCommand(MOVETO, 1), BitUtil.zigZagEncode(minX), BitUtil.zigZagEncode(minY),
            encodeCommand(LINETO, 3),
            BitUtil.zigZagEncode(0), BitUtil.zigZagEncode(maxY - minY),
            BitUtil.zigZagEncode(maxX - minX), BitUtil.zigZagEncode(0),
            BitUtil.zigZagEncode(0), BitUtil.zigZagEncode(minY - maxY),
            encodeCommand(CLOSEPATH, 1)};
        return writeCommands(commands, 3, 11);
    }
}