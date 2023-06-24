package org.elasticsearch.xpack.spatial.index.fielddata;

public interface TriangleTreeVisitor {
    
    //Visits a node point.
    void visitPoint(int x, int y);

    //Visits a node line.
    void visitLine(int aX, int aY, int bX, int bY, byte metadata);

    //Visits a node triangle.
    void visitTriangle(int aX, int aY, int bX, int bY, int cX, int cY, byte metadata);

    //Determines if the visitor should keep visiting the tree.
    boolean push();

    //Determines if the visitor should visit nodes with bounds greater or equal to minX.
    boolean pushX(int minX);

    //Determines if the visitor should visit nodes with bounds greater or equal to minY.
    boolean pushY(int minY);

    //Determines if the visitor should visit nodes with bounds lower or equal than maxX and maxY.
    boolean push(int maxX, int maxY);

    //Determines if the visitor should visit the tree given the bounding box of the tree.
    boolean push(int minX, int minY, int maxX, int maxY);

    abstract class TriangleTreeDecodedVisitor implements TriangleTreeVisitor {

        private final CoordinateEncoder encoder;

        protected TriangleTreeDecodedVisitor(CoordinateEncoder encoder) {
            this.encoder = encoder;
        }

        //Visits a decoded point.
        protected abstract void visitDecodedPoint(double x, double y);

        //Visits a decoded line.
        protected abstract void visitDecodedLine(double aX, double aY, double bX, double bY, byte metadata);

        //Visits a decoded triangle.
        protected abstract void visitDecodedTriangle(double aX, double aY, double bX, double bY, double cX, double cY, byte metadata);

        //Determines if the visitor should visit nodes with bounds greater or equal to decoded minX.
        protected abstract boolean pushDecodedX(double minX);

        //Determines if the visitor should visit nodes with bounds greater or equal to decoded minY.
        protected abstract boolean pushDecodedY(double minX);

        //Determines if the visitor should visit nodes with bounds lower or equal than decoded maxX and maxY.
        protected abstract boolean pushDecoded(double maxX, double maxY);

        //Determines if the visitor should visit the tree given the bounding box of the tree in decoded form.
        protected abstract boolean pushDecoded(double minX, double minY, double maxX, double maxY);

        @Override
        public final void visitPoint(int x, int y) {
            visitDecodedPoint(encoder.decodeX(x), encoder.decodeY(y));
        }

        @Override
        public final void visitLine(int aX, int aY, int bX, int bY, byte metadata) {
            visitDecodedLine(encoder.decodeX(aX), encoder.decodeY(aY), encoder.decodeX(bX), encoder.decodeY(bY), metadata);
        }

        @Override
        public final void visitTriangle(int aX, int aY, int bX, int bY, int cX, int cY, byte metadata) {
            visitDecodedTriangle(encoder.decodeX(aX), encoder.decodeY(aY), encoder.decodeX(bX), encoder.decodeY(bY), encoder.decodeX(cX), encoder.decodeY(cY), metadata);
        }

        @Override
        public final boolean pushX(int minX) {
            return pushDecodedX(encoder.decodeX(minX));
        }

        @Override
        public final boolean pushY(int minY) {
            return pushDecodedY(encoder.decodeY(minY));
        }

        @Override
        public final boolean push(int maxX, int maxY) {
            return pushDecoded(encoder.decodeX(maxX), encoder.decodeY(maxY));
        }

        @Override
        public final boolean push(int minX, int minY, int maxX, int maxY) {
            return pushDecoded(encoder.decodeX(minX), encoder.decodeY(minY), encoder.decodeX(maxX), encoder.decodeY(maxY));
        }
    }
}