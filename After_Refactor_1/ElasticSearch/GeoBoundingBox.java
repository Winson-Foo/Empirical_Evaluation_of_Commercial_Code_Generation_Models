package org.elasticsearch.common.geo;

import java.io.IOException;

import org.elasticsearch.ElasticsearchParseException;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.xcontent.ParseField;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentParser;

/**
 * A class representing a Geo-Bounding-Box for use by Geo queries and aggregations
 * that deal with extents/rectangles representing rectangular areas of interest.
 */
public class GeoBoundingBox extends BoundingBox<GeoPoint> {

    public static final ParseField LAT_FIELD = new ParseField("lat");
    public static final ParseField LON_FIELD = new ParseField("lon");
    public static final String TOP_LEFT_NAME = "top_left";
    public static final String BOTTOM_RIGHT_NAME = "bottom_right";

    public GeoBoundingBox(GeoPoint topLeft, GeoPoint bottomRight) {
        super(topLeft, bottomRight);
    }

    public GeoBoundingBox(StreamInput input) throws IOException {
        super(input.readGeoPoint(), input.readGeoPoint());
    }

    /**
     * Returns the top left {@link GeoPoint} of the bounding box.
     *
     * @return the top left {@link GeoPoint} of the bounding box.
     */
    @Override
    public GeoPoint topLeft() {
        return topLeft;
    }

    /**
     * Returns the bottom right {@link GeoPoint} of the bounding box.
     *
     * @return the bottom right {@link GeoPoint} of the bounding box.
     */
    @Override
    public GeoPoint bottomRight() {
        return bottomRight;
    }

    /**
     * Builds an {@link XContentBuilder} representation of the bounding box with keyed map format.
     *
     * @param builder the builder to use for building the XContent fragment
     * @return the builder with added XContent for the bounding box.
     * @throws IOException in case of an I/O error during XContent building.
     */
    public XContentBuilder toXContentWithMap(XContentBuilder builder) throws IOException {
        builder.startObject(TOP_LEFT_NAME);
        builder.field(LAT_FIELD.getPreferredName(), topLeft.getY());
        builder.field(LON_FIELD.getPreferredName(), topLeft.getX());
        builder.endObject();
        builder.startObject(BOTTOM_RIGHT_NAME);
        builder.field(LAT_FIELD.getPreferredName(), bottomRight.getY());
        builder.field(LON_FIELD.getPreferredName(), bottomRight.getX());
        builder.endObject();
        return builder;
    }

    /**
     * Builds an {@link XContentBuilder} representation of the bounding box with array format.
     * This format is used specifically for the GeoBoundingBoxQueryBuilder.
     *
     * @param builder the builder to use for building the XContent fragment
     * @return the builder with added XContent for the bounding box.
     * @throws IOException in case of an I/O error during XContent building.
     */
    public XContentBuilder toXContentWithArray(XContentBuilder builder) throws IOException {
        builder.array(TOP_LEFT_NAME, topLeft.getX(), topLeft.getY());
        builder.array(BOTTOM_RIGHT_NAME, bottomRight.getX(), bottomRight.getY());
        return builder;
    }

    /**
     * Determines whether the point (lon, lat) falls within the specified bounding box.
     *
     * @param lon the longitude of the point
     * @param lat the latitude of the point
     * @return true if the point (lon, lat) is within the bounding box, false otherwise.
     */
    public boolean pointInBounds(double lon, double lat) {
        if (lat >= bottom() && lat <= top()) {
            if (left() <= right()) {
                return lon >= left() && lon <= right();
            } else {
                return lon >= left() || lon <= right();
            }
        }
        return false;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeGeoPoint(topLeft);
        out.writeGeoPoint(bottomRight);
    }

    /**
     * A parser for parsing bounding box coordinates from a {@link XContentParser}.
     */
    protected static class GeoBoundsParser extends BoundsParser<GeoBoundingBox> {

        public GeoBoundsParser(XContentParser parser) {
            super(parser);
        }

        @Override
        protected GeoBoundingBox createWithEnvelope() {
            GeoPoint topLeft = new GeoPoint(envelope.getMaxLat(), envelope.getMinLon());
            GeoPoint bottomRight = new GeoPoint(envelope.getMinLat(), envelope.getMaxLon());
            return new GeoBoundingBox(topLeft, bottomRight);
        }

        @Override
        protected GeoBoundingBox createWithBounds() {
            GeoPoint topLeft = new GeoPoint(top, left);
            GeoPoint bottomRight = new GeoPoint(bottom, right);
            return new GeoBoundingBox(topLeft, bottomRight);
        }

        @Override
        protected SpatialPoint parsePointWith(XContentParser parser, GeoUtils.EffectivePoint effectivePoint) throws IOException, ElasticsearchParseException {
            return GeoUtils.parseGeoPoint(parser, false, effectivePoint);
        }
    }

    /**
     * Parses the bounding box and returns bottom, top, left, right coordinates.
     *
     * @param parser the parser to use for parsing the bounding box.
     * @return the bounding box parsed from the input parser.
     * @throws IOException in case of I/O error during parsing
     * @throws ElasticsearchParseException in case of an error during parsing.
     */
    public static GeoBoundingBox parseBoundingBox(XContentParser parser) throws IOException, ElasticsearchParseException {
        GeoBoundsParser bounds = new GeoBoundsParser(parser);
        return bounds.parseBoundingBox();
    }
}