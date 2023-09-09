package org.elasticsearch.legacygeo;

import org.locationtech.spatial4j.context.SpatialContext;
import org.locationtech.spatial4j.shape.Shape;
import org.locationtech.spatial4j.shape.ShapeCollection;

import java.util.List;

/**
 * A custom shape collection that supports points only indexing
 */
public class PointOnlyShapeCollection<S extends Shape> extends ShapeCollection<S> {

    /**
     * Whether this collection should only index points
     */
    private boolean pointsOnly;

    public PointOnlyShapeCollection(List<S> shapes, SpatialContext ctx) {
        super(shapes, ctx);
    }

    /**
     * Checks if this collection only indexes points
     *
     * @return true if only points are being indexed
     */
    public boolean isPointsOnly() {
        return pointsOnly;
    }

    /**
     * Sets whether this collection should only index points
     *
     * @param pointsOnly true to only index points, false otherwise
     */
    public void setPointsOnly(boolean pointsOnly) {
        this.pointsOnly = pointsOnly;
    }
}
