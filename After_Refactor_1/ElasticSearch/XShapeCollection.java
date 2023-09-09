package org.elasticsearch.legacygeo;

import org.locationtech.spatial4j.context.SpatialContext;
import org.locationtech.spatial4j.shape.Point;
import org.locationtech.spatial4j.shape.ShapeCollection;

import java.util.List;

/**
 * A ShapeCollection that supports only point shapes.
 */
public class PointShapeCollection extends ShapeCollection<Point> {

    private boolean pointsOnly = true;

    public PointShapeCollection(List<Point> shapes, SpatialContext ctx) {
        super(shapes, ctx);
    }

    /**
     * Returns whether this collection contains only point shapes.
     */
    public boolean isPointsOnly() {
        return this.pointsOnly;
    }
}