package org.elasticsearch.legacygeo;

import org.locationtech.spatial4j.context.SpatialContext;
import org.locationtech.spatial4j.shape.Shape;
import org.locationtech.spatial4j.shape.ShapeCollection;

import java.util.List;

/**
 * This class extends the Spatial4j ShapeCollection class to add support for points-only shape indexing.
 */
public class XShapeCollection<S extends Shape> extends ShapeCollection<S> {

    private boolean isPointsOnly = false;

    public XShapeCollection(List<S> shapes, SpatialContext context) {
        super(shapes, context);
    }

    /**
     * Returns true if shape indexing is limited to points only, false otherwise.
     */
    public boolean isPointsOnly() {
        return this.isPointsOnly;
    }

    /**
     * Sets the shape indexing mode to points only if the provided flag is true.
     *
     * @param isPointsOnly - the flag indicating whether to limit shape indexing to points only.
     */
    public void setPointsOnly(boolean isPointsOnly) {
        this.isPointsOnly = isPointsOnly;
    }
}