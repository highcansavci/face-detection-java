package com.savci.facedetection.entity;

import com.savci.facedetection.model.FaceAnnotation;

public class FaceDetectionResult {
    private FaceAnnotation[] faces;
    private int totalFaces;

    // Getters and setters
    public FaceAnnotation[] getFaces() {
        return faces;
    }

    public void setFaces(FaceAnnotation[] faces) {
        this.faces = faces;
    }

    public int getTotalFaces() {
        return totalFaces;
    }

    public void setTotalFaces(int totalFaces) {
        this.totalFaces = totalFaces;
    }
}