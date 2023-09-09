package org.elasticsearch.xpack.spatial.index.fielddata;

public interface TriangleTreeVisitor {

    void visitPoint(int x, int y);

    void visitLine(int aX, int aY, int bX, int bY, byte metadata);

    void visitTriangle(int aX, int aY, int bX, int bY, int cX, int cY, byte metadata);

    boolean shouldContinue();

    boolean shouldVisitX(int minX);

    boolean shouldVisitY(int minY);

    boolean shouldVisit(int maxX, int maxY);

    boolean shouldVisitTree(int minX, int minY, int maxX, int maxY);

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
        public final void visitLine(int aX, int aY, int bX, int bY, byte metadata) {
            visitDecodedLine(encoder.decodeX(aX), encoder.decodeY(aY), encoder.decodeX(bX), encoder.decodeY(bY), metadata);
        }

        protected abstract void visitDecodedLine(double aX, double aY, double bX, double bY, byte metadata);

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
        public final boolean shouldVisitX(int minX) {
            return shouldVisitDecodedX(encoder.decodeX(minX));
        }

        protected abstract boolean shouldVisitDecodedX(double minX);

        @Override
        public final boolean shouldVisitY(int minY) {
            return shouldVisitDecodedY(encoder.decodeY(minY));
        }

        protected abstract boolean shouldVisitDecodedY(double minX);

        @Override
        public final boolean shouldVisit(int maxX, int maxY) {
            return shouldVisitDecoded(encoder.decodeX(maxX), encoder.decodeY(maxY));
        }

        protected abstract boolean shouldVisitDecoded(double maxX, double maxY);

        @Override
        public final boolean shouldVisitTree(int minX, int minY, int maxX, int maxY) {
            return shouldVisitDecoded(encoder.decodeX(minX), encoder.decodeY(minY), encoder.decodeX(maxX), encoder.decodeY(maxY));
        }

        protected abstract boolean shouldVisitDecoded(double minX, double minY, double maxX, double maxY);
    }
}