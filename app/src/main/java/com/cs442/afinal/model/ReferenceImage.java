package com.cs442.afinal.model;

import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;

/**
 * Created by stymjs0515 on 26/05/2018.
 */

public class ReferenceImage {

    private Mat gray;
    private Mat rgba;
    private Mat descriptors;
    private MatOfKeyPoint keyPoints;

    public ReferenceImage() {
    }

    public ReferenceImage(Mat gray, Mat rgba, Mat descriptors, MatOfKeyPoint keyPoints) {
        this.gray = gray;
        this.rgba = rgba;
        this.descriptors = descriptors;
        this.keyPoints = keyPoints;
    }

    public Mat getGray() {
        return gray;
    }

    public void setGray(Mat gray) {
        this.gray = gray;
    }

    public Mat getRgba() {
        return rgba;
    }

    public void setRgba(Mat rgba) {
        this.rgba = rgba;
    }

    public Mat getDescriptors() {
        return descriptors;
    }

    public void setDescriptors(Mat descriptors) {
        this.descriptors = descriptors;
    }

    public MatOfKeyPoint getKeyPoints() {
        return keyPoints;
    }

    public void setKeyPoints(MatOfKeyPoint keyPoints) {
        this.keyPoints = keyPoints;
    }
}
