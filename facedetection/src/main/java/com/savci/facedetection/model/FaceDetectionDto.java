package com.savci.facedetection.model;

import org.nd4j.linalg.api.ndarray.INDArray;

public class FaceDetectionDto {
    
    private final INDArray dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph;

    private FaceDetectionDto(Builder builder) {
        this.dy = builder.dy;
        this.edy = builder.edy;
        this.dx = builder.dx;
        this.edx = builder.edx;
        this.y = builder.y;
        this.ey = builder.ey;
        this.x = builder.x;
        this.ex = builder.ex;
        this.tmpw = builder.tmpw;
        this.tmph = builder.tmph;
    }

    // Getters
    public INDArray getDy() {
        return dy;
    }

    public INDArray getEdy() {
        return edy;
    }

    public INDArray getDx() {
        return dx;
    }

    public INDArray getEdx() {
        return edx;
    }

    public INDArray getY() {
        return y;
    }

    public INDArray getEy() {
        return ey;
    }

    public INDArray getX() {
        return x;
    }

    public INDArray getEx() {
        return ex;
    }

    public INDArray getTmpw() {
        return tmpw;
    }

    public INDArray getTmph() {
        return tmph;
    }

    // Builder Class
    public static class Builder {
        private INDArray dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph;

        public Builder() {
        }

        public Builder dy(INDArray dy) {
            this.dy = dy;
            return this;
        }

        public Builder edy(INDArray edy) {
            this.edy = edy;
            return this;
        }

        public Builder dx(INDArray dx) {
            this.dx = dx;
            return this;
        }

        public Builder edx(INDArray edx) {
            this.edx = edx;
            return this;
        }

        public Builder y(INDArray y) {
            this.y = y;
            return this;
        }

        public Builder ey(INDArray ey) {
            this.ey = ey;
            return this;
        }

        public Builder x(INDArray x) {
            this.x = x;
            return this;
        }

        public Builder ex(INDArray ex) {
            this.ex = ex;
            return this;
        }

        public Builder tmpw(INDArray tmpw) {
            this.tmpw = tmpw;
            return this;
        }

        public Builder tmph(INDArray tmph) {
            this.tmph = tmph;
            return this;
        }

        public FaceDetectionDto build() {
            return new FaceDetectionDto(this);
        }
    }
}
