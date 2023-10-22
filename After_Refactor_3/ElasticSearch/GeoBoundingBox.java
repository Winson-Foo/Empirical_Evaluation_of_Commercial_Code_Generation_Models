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
    private static final String LAT_FIELD_NAME = "lat";
    private static final String LON_FIELD_NAME = "lon";
    private static final String TOP_LEFT_FIELD_NAME = "top_left";
    private static final String BOTTOM_RIGHT_FIELD_NAME = "bottom_right";
    private static final double MIN_LATITUDE = -90.0;
    private static final double MAX_LATITUDE = 90.0;
    private static final double MIN_LONGITUDE = -180.0;
    private static final double MAX_LONGITUDE = 180.0;

    @ParseField(TOP_LEFT_FIELD_NAME)
    private final GeoPoint topLeft;
    @ParseField(BOTTOM_RIGHT_FIELD_NAME)
    private final GeoPoint bottomRight;

    private GeoBoundingBox(GeoPoint topLeft, GeoPoint bottomRight) {
        super(topLeft, bottomRight);
        this.topLeft = topLeft;
        this.bottomRight = bottomRight;
    }

    public GeoPoint getTopLeft() {
        return topLeft;
    }

    public GeoPoint getBottomRight() {
        return bottomRight;
    }

    public boolean containsPoint(GeoPoint point) {
        double latitude = point.getY();
        double longitude = point.getX();
        
        boolean withinLatitudeBounds = latitude >= bottom() && latitude <= top();
        boolean withinLongitudeBounds = left() <= right() ? longitude >= left() && longitude <= right()
                                                         : longitude >= left() || longitude <= right();

        return withinLatitudeBounds && withinLongitudeBounds;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeGeoPoint(topLeft);
        out.writeGeoPoint(bottomRight);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        if (params.paramAsBoolean("use_array_format", false)) {
            builder.startArray(TOP_LEFT_FIELD_NAME).value(topLeft.getX()).value(topLeft.getY()).endArray();
            builder.startArray(BOTTOM_RIGHT_FIELD_NAME).value(bottomRight.getX()).value(bottomRight.getY()).endArray();
        } else {
            builder.startObject(TOP_LEFT_FIELD_NAME).field(LAT_FIELD_NAME, topLeft.getY()).field(LON_FIELD_NAME, topLeft.getX()).endObject();
            builder.startObject(BOTTOM_RIGHT_FIELD_NAME).field(LAT_FIELD_NAME, bottomRight.getY()).field(LON_FIELD_NAME, bottomRight.getX()).endObject();
        }
        return builder;
    }

    public static class Builder {
        private GeoPoint topLeft;
        private GeoPoint bottomRight;

        public Builder setTopLeft(GeoPoint topLeft) {
            this.topLeft = topLeft;
            return this;
        }

        public Builder setBottomRight(GeoPoint bottomRight) {
            this.bottomRight = bottomRight;
            return this;
        }

        public GeoBoundingBox build() {
            validateParameters(topLeft, bottomRight);
            return new GeoBoundingBox(topLeft, bottomRight);
        }

        private void validateParameters(GeoPoint topLeft, GeoPoint bottomRight) {
            if (topLeft.getY() < MIN_LATITUDE || topLeft.getY() > MAX_LATITUDE
                    || bottomRight.getY() < MIN_LATITUDE || bottomRight.getY() > MAX_LATITUDE
                    || topLeft.getX() < MIN_LONGITUDE || topLeft.getX() > MAX_LONGITUDE
                    || bottomRight.getX() < MIN_LONGITUDE || bottomRight.getX() > MAX_LONGITUDE) {
                throw new IllegalArgumentException("Invalid bounding box coordinates");
            }
        }
    }

    public static GeoBoundingBox fromStream(StreamInput input) throws IOException {
        GeoPoint topLeft = input.readGeoPoint();
        GeoPoint bottomRight = input.readGeoPoint();
        return new GeoBoundingBox(topLeft, bottomRight);
    }

    public static GeoBoundingBox fromXContent(XContentParser parser) throws IOException, ElasticsearchParseException {
        double top = Double.NaN, left = Double.NaN, bottom = Double.NaN, right = Double.NaN;

        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken(); // move to the value

            switch (fieldName) {
                case LAT_FIELD_NAME:
                    double latitude = parser.doubleValue();
                    if (Double.isNaN(top)) {
                        top = latitude;
                    } else {
                        bottom = latitude;
                    }
                    break;

                case LON_FIELD_NAME:
                    double longitude = parser.doubleValue();
                    if (Double.isNaN(left)) {
                        left = longitude;
                    } else {
                        right = longitude;
                    }
                    break;

                default:
                    throw new ElasticsearchParseException("Unexpected field: " + fieldName);
            }
        }

        if (Double.isNaN(top) || Double.isNaN(left) || Double.isNaN(bottom) || Double.isNaN(right)) {
            throw new ElasticsearchParseException("Invalid bounding box format: " + parser.text());
        }

        Builder builder = new Builder();
        builder.setTopLeft(new GeoPoint(top, left));
        builder.setBottomRight(new GeoPoint(bottom, right));
        return builder.build();
    }
}