package org.elasticsearch.xpack.spatial.index.fielddata;

public interface TriangleTreeVisitor {

    void visitPoint(int x, int y);

    void visitLine(int startX, int startY, int endX, int endY, byte metadata);

    void visitTriangle(int aX, int aY, int bX, int bY, int cX, int cY, byte metadata);

    boolean continueVisit();

    boolean continueVisitFromX(int minX);

    boolean continueVisitFromY(int minY);

    boolean continueVisitUntil(int maxX, int maxY);

    boolean continueVisitWithinBoundingBox(int minX, int minY, int maxX, int maxY);

    abstract class TriangleTreeDecodedVisitor implements TriangleTreeVisitor {

        private final CoordinateEncoder encoder;

        protected TriangleTreeDecodedVisitor(CoordinateEncoder encoder) {
            this.encoder = encoder;
        }

        @Override
        public final void visitPoint(int x, int y) {
            visitDecodedPoint(encoder.decodeX(x), encoder.decodeY(y));
        }

        protected abstract void visitDecodedPoint(double x, double y);

        @Override
        public final void visitLine(int startX, int startY, int endX, int endY, byte metadata) {
            visitDecodedLine(encoder.decodeX(startX), encoder.decodeY(startY), encoder.decodeX(endX), encoder.decodeY(endY), metadata);
        }

        protected abstract void visitDecodedLine(double startX, double startY, double endX, double endY, byte metadata);

        @Override
        public final void visitTriangle(int aX, int aY, int bX, int bY, int cX, int cY, byte metadata) {
            visitDecodedTriangle(
                encoder.decodeX(aX),
                encoder.decodeY(aY),
                encoder.decodeX(bX),
                encoder.decodeY(bY),
                encoder.decodeX(cX),
                encoder.decodeY(cY),
                metadata
            );
        }

        protected abstract void visitDecodedTriangle(double aX, double aY, double bX, double bY, double cX, double cY, byte metadata);

        @Override
        public final boolean continueVisitFromX(int minX) {
            return continueVisitDecodedFromX(encoder.decodeX(minX));
        }

        protected abstract boolean continueVisitDecodedFromX(double minX);

        @Override
        public final boolean continueVisitFromY(int minY) {
            return continueVisitDecodedFromY(encoder.decodeY(minY));
        }

        protected abstract boolean continueVisitDecodedFromY(double minY);

        @Override
        public final boolean continueVisitUntil(int maxX, int maxY) {
            return continueVisitDecodedUntil(encoder.decodeX(maxX), encoder.decodeY(maxY));
        }

        protected abstract boolean continueVisitDecodedUntil(double maxX, double maxY);

        @Override
        public final boolean continueVisitWithinBoundingBox(int minX, int minY, int maxX, int maxY) {
            return continueVisitDecodedWithinBoundingBox(encoder.decodeX(minX), encoder.decodeY(minY), encoder.decodeX(maxX), encoder.decodeY(maxY));
        }

        protected abstract boolean continueVisitDecodedWithinBoundingBox(double minX, double minY, double maxX, double maxY);
    }
}