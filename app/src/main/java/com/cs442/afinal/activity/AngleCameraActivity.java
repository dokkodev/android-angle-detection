package com.cs442.afinal.activity;

import android.graphics.Bitmap;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;

import com.cs442.afinal.utils.MatrixMultiplication;
import com.cs442.afinal.R;
import com.cs442.afinal.model.CameraFrame;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.ORB;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;

public class AngleCameraActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2, SensorEventListener {

    private static final String TAG = AngleCameraActivity.class.getName();

    private static int frameCounter = 0;

    private CameraBridgeViewBase mOpenCvCameraView;
    private SensorManager mSensorManager;
    private ImageView matchImage;

    Mat cameraIntrinsic;

    private double sumAngle = 9999;
    private double lastAngleDiff = 0;

    TextView magneticAngle;
    TextView imageAngle;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_camera);

        mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);

        // TextView that will tell the user what degree is he heading
        magneticAngle = (TextView) findViewById(R.id.magnetic_angle);
        imageAngle = (TextView) findViewById(R.id.image_angle);

        mOpenCvCameraView = findViewById(R.id.CvCameraView);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);

        mSensorManager.registerListener(this,
                mSensorManager.getDefaultSensor(Sensor.TYPE_ORIENTATION),
                SensorManager.SENSOR_DELAY_GAME);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        mSensorManager.unregisterListener(this);

        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        //return inputFrame.rgba();
        return calculateAngleDisplacement(inputFrame);
    }

    int framePerCalc = 12;
    double matcherThreshold = 0.70;
    CameraFrame tmpCameraFrame;
    int nothingCounter;

    CameraFrame refCameraFrame;
    CameraFrame newCameraFrame;
    List<DMatch> goodMatches;

    private Mat calculateAngleDisplacement(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        if (goodMatches != null) {
            goodMatches.clear();
        }
        /*
         *  1. Check Frame Counter
         */
        frameCounter++;
        if (frameCounter < framePerCalc) {
            //return bufferedOutputImage;
            return inputFrame.rgba();
        }
        frameCounter = 0;

        Mat newRgba = inputFrame.rgba().clone();

        /*
         *  2. Orb Feature detect
         */
        orbFeatureDetect(newRgba);

        /*
         *  3. Check if previous frame exists
         *  Cannot do matching so return
         */
        if (refCameraFrame == null) {
            refCameraFrame = newCameraFrame;
            return newRgba;
        }

        /*
         *  4. Descriptor Matching
         *  Use extended for symmetric matches
         */
        matchDescriptors();
        //extendedMatchDescriptors();
        if (goodMatches.size() < 5) {
            nothingCounter++;
            if (nothingCounter > 2) {
                refCameraFrame = tmpCameraFrame;
                sumAngle -= lastAngleDiff;
                nothingCounter = 0;
            }
            return newRgba;
        }

        /*
         *  5. Find Homography Matrix
         */
        findHomography();

        /*
         *  6. Do something with picture
         */
        if (goodMatches.size() > 10) {
            Scalar redColor = new Scalar(255, 0, 0);
            Scalar greenColor = new Scalar(0, 255, 0);
            Scalar allColor = Scalar.all(-1);

            Mat clonedRefRGBA = new Mat();
            Mat clonedNewRGBA = new Mat();
            Imgproc.cvtColor(refCameraFrame.getRgba(), clonedRefRGBA, Imgproc.COLOR_RGBA2RGB, 3);
            Imgproc.cvtColor(newCameraFrame.getRgba(), clonedNewRGBA, Imgproc.COLOR_RGBA2RGB, 3);

            Mat outputImage = new Mat();
            MatOfByte matchLines = new MatOfByte();
            MatOfDMatch goodMatOfDMatch = new MatOfDMatch();

            List<DMatch> drawMatches = goodMatches;
            Collections.sort(goodMatches, new Comparator<DMatch>() {
                @Override
                public int compare(DMatch o1, DMatch o2) {
                    return (int) (o2.distance - o1.distance);
                }
            });

            drawMatches = goodMatches.subList(0, Math.min(50, goodMatches.size() - 1));
            goodMatOfDMatch.fromList(drawMatches);


            Features2d.drawMatches(
                    clonedNewRGBA, newCameraFrame.getKeyPoints(),
                    clonedRefRGBA, refCameraFrame.getKeyPoints(),

                    goodMatOfDMatch, outputImage, Scalar.all(-1), Scalar.all(-1), matchLines,
                    Features2d.NOT_DRAW_SINGLE_POINTS);

            //Imgproc.resize(outputImage, outputImage, inputFrame.rgba().size());

            // convert to bitmap:
            final Bitmap bm = Bitmap.createBitmap(outputImage.cols(), outputImage.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(outputImage, bm);

            matchImage = (ImageView) findViewById(R.id.match);
            // find the imageview and draw it!
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    matchImage.setImageBitmap(bm);
                }
            });
        }

        // Puts the new image to reference and return
        tmpCameraFrame = refCameraFrame;
        refCameraFrame = newCameraFrame;
        newCameraFrame = new CameraFrame();

        //bufferedOutputImage = outputImage;

        return inputFrame.rgba();
    }

    ORB orbFeature;
    DescriptorMatcher matcher;

    private void init() {
        orbFeature = ORB.create(500, 1.2f, 8, 31,
                0, 2, ORB.HARRIS_SCORE, 31, 20);
        matcher = BFMatcher.create(Core.NORM_HAMMING, false);
    }

    private void orbFeatureDetect(Mat cameraFrameRgba) {
        // detect key points
        MatOfKeyPoint keyPoints = new MatOfKeyPoint();
        orbFeature.detect(cameraFrameRgba, keyPoints);

        // compute descriptors for each key points
        Mat descriptors = new Mat();
        orbFeature.compute(cameraFrameRgba, keyPoints, descriptors);

        newCameraFrame = new CameraFrame();
        newCameraFrame.setKeyPoints(keyPoints);
        newCameraFrame.setDescriptors(descriptors);
        newCameraFrame.setRgba(cameraFrameRgba);
    }

    private void matchDescriptors() {

        List<MatOfDMatch> matches = new ArrayList<>();

        Mat refDescriptors = refCameraFrame.getDescriptors();
        Mat newDescriptors = newCameraFrame.getDescriptors();

        matcher.knnMatch(newDescriptors, refDescriptors, matches, 2);
        Log.d(TAG, String.format("Number of matches : %d", matches.size()));

        KeyPoint[] refKeyPoints = refCameraFrame.getKeyPoints().toArray();
        KeyPoint[] newKeyPoints = newCameraFrame.getKeyPoints().toArray();

//        ArrayList<Double> distances = new ArrayList<>();
//        double sum = 0;
//        for (int i = 0; i < matches.size(); i++) {
//            DMatch bestMatch = matches.get(i).toArray()[0];
//            double rx = refKeyPoints[bestMatch.trainIdx].pt.x;
//            double ry = refKeyPoints[bestMatch.trainIdx].pt.y;
//            double nx = newKeyPoints[bestMatch.queryIdx].pt.x;
//            double ny = newKeyPoints[bestMatch.queryIdx].pt.y;
//            double dist = Math.hypot(rx - nx, ry - ny);
//            //Log.d(TAG, String.format("(%f, %f), (%f, %f), %f", rx, ry, nx, ny, dist));
//            sum += dist;
//            distances.add(dist);
//        }
//        double avg = sum / matches.size();
//        Log.d(TAG, String.valueOf(avg));
//
//        Collections.sort(distances);
//        Log.d(TAG, distances.toString());


        // Filter good matches by ratio test
        LinkedList<DMatch> filteredMatches = new LinkedList<>();
        for (int i = 0; i < matches.size(); i++) {
            DMatch bestMatch = matches.get(i).toArray()[0];
            DMatch nextMatch = matches.get(i).toArray()[1];
//
//            double rx = refKeyPoints[bestMatch.trainIdx].pt.x;
//            double ry = refKeyPoints[bestMatch.trainIdx].pt.y;
//            double nx = newKeyPoints[bestMatch.queryIdx].pt.x;
//            double ny = newKeyPoints[bestMatch.queryIdx].pt.y;
//            double dist = Math.hypot(rx - nx, ry - ny);
//
//            if (dist > 170) {
//                continue;
//            }

            if (bestMatch.distance > 25) {
                continue;
            }

            if (bestMatch.distance / nextMatch.distance < matcherThreshold) {
                filteredMatches.add(bestMatch);
            }
        }

        Log.d(TAG, String.format("Number of good matches : %d", filteredMatches.size()));

        goodMatches = filteredMatches;
    }

    private void extendedMatchDescriptors() {

        Mat refDescriptors = refCameraFrame.getDescriptors();
        Mat newDescriptors = newCameraFrame.getDescriptors();

        List<MatOfDMatch> matches = new ArrayList<>();
        matcher.knnMatch(refDescriptors, newDescriptors, matches, 2);
        Log.d(TAG, String.format("Number of matches the other way : %d", matches.size()));

        // Filter good matches by ratio test
        LinkedList<DMatch> filteredMatches = new LinkedList<>();
        for (int i = 0; i < matches.size(); i++) {
            DMatch bestMatch = matches.get(i).toArray()[0];
            DMatch nextMatch = matches.get(i).toArray()[1];

            if (bestMatch.distance > 20) {
                continue;
            }

            if (bestMatch.distance / nextMatch.distance < matcherThreshold) {
                filteredMatches.add(bestMatch);
            }
        }

        // Only Symmetric matches
        LinkedList<DMatch> symmetricMatches = new LinkedList<>();
        for (int i = 0; i < goodMatches.size(); i++) {
            for (int j = 0; j < filteredMatches.size(); j++) {
                if (goodMatches.get(i).queryIdx == filteredMatches.get(j).trainIdx
                        && goodMatches.get(i).trainIdx == filteredMatches.get(j).queryIdx) {
                    symmetricMatches.addLast(goodMatches.get(i));
                }
            }
        }

        Log.d(TAG, String.format("Number of good matches with symmetric : %d", goodMatches.size()));

        goodMatches = symmetricMatches;
    }

    private void findHomography() {

        LinkedList<Point> refPoints = new LinkedList<>();
        LinkedList<Point> newPoints = new LinkedList<>();
        List<KeyPoint> refKeyPoints = refCameraFrame.getKeyPoints().toList();
        List<KeyPoint> newKeyPoints = newCameraFrame.getKeyPoints().toList();

        List<KeyPoint> tmpKeyPoints = new LinkedList<>();

        for (int i = 0; i < goodMatches.size(); i++) {
            refPoints.addLast(refKeyPoints.get(goodMatches.get(i).trainIdx).pt);
            newPoints.addLast(newKeyPoints.get(goodMatches.get(i).queryIdx).pt);
            tmpKeyPoints.add(newKeyPoints.get(goodMatches.get(i).queryIdx));
        }

        MatOfPoint2f refMatPoints = new MatOfPoint2f();
        MatOfPoint2f newMatPoints = new MatOfPoint2f();
        refMatPoints.fromList(refPoints);
        newMatPoints.fromList(newPoints);

        Mat homography = Calib3d.findHomography(refMatPoints, newMatPoints, Calib3d.FM_RANSAC, 3);
        //(TAG, hg.dump());

        if (homography.cols() > 0) {
            getAveragePixelDisplacement(homography);
        }
    }

    private double get(Mat mat, int row, int col) {
        return mat.get(row, col)[0];
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    initializeIntrinsicCameraMat();
                    init();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    float frameWidth;
    float frameHeight;
    float hVA;
    float vVA;

    private void initializeIntrinsicCameraMat() {
        frameWidth = mOpenCvCameraView.getmFrameWidth();
        frameHeight = mOpenCvCameraView.getmFrameHeight();
        hVA = mOpenCvCameraView.getmHorizontalViewAngle();
        vVA = mOpenCvCameraView.getmVerticalViewAngle();

        Log.d(TAG, String.valueOf(frameWidth));
        Log.d(TAG, String.valueOf(frameHeight));
        Log.d(TAG, String.valueOf(hVA));
        Log.d(TAG, String.valueOf(vVA));

        Mat intrinsics = Mat.eye(3, 3, CvType.CV_32F);
        intrinsics.put(0, 0, frameWidth / Math.tan(Math.toRadians(hVA / 2)));
        intrinsics.put(1, 1, frameHeight / Math.tan(Math.toRadians(vVA / 2)));
        intrinsics.put(0, 2, frameWidth / 2);
        intrinsics.put(1, 2, frameHeight / 2);

        Log.d(TAG, intrinsics.dump());

        cameraIntrinsic = intrinsics;
    }

    private void getAveragePixelDisplacement(Mat hg) {

        double[][] homography = new double[][]
                {
                        {get(hg, 0, 0), get(hg, 0, 1), get(hg, 0, 2)},
                        {get(hg, 1, 0), get(hg, 1, 1), get(hg, 1, 2)},
                        {get(hg, 2, 0), get(hg, 2, 1), get(hg, 2, 2)}
                };


        int average = getAverage(homography, 2, 6);
        double angle = Math.atan(2 * average * Math.tan(Math.toRadians(mOpenCvCameraView.getmHorizontalViewAngle())) / frameWidth);
        //Log.d(TAG, "angle : " + Math.toDegrees(angle));
        lastAngleDiff = Math.toDegrees(angle);
        if (sumAngle == 9999) {
            sumAngle = magneticSumAngle;
        }
        sumAngle += Math.toDegrees(angle);
        if (sumAngle >= 360)
            sumAngle -= 360;
        if (sumAngle < 0)
            sumAngle += 360;
        Log.d(TAG, "sum angle : " + String.format("%.2f", sumAngle));

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                imageAngle.setText(String.format("Image angle : %s", String.format("%.1f", sumAngle)));
            }
        });
    }

    private int getHorizontalDisplacement(double[][] homography, double x, double y) {


        //Log.d(TAG, String.valueOf(frameWidth));

        double[][] point = new double[][] {{frameWidth * x}, {frameHeight * y}, {1}};

        double[][] result = MatrixMultiplication.multiply(homography, point);
        double newXCoord = result[0][0];


        //Log.d(TAG, Arrays.deepToString(point) + " " + Arrays.deepToString(result));
        int ret = (int) (frameWidth * x - newXCoord);
        //Log.d(TAG, "displacement : " + String.valueOf(ret));
        return ret;
    }

    private int getAverage(double[][] homography, int low, int high) {

        double lowFr = (float) low / 8;
        double midFr = (float) 4 / 8;
        double higFr = (float) high / 8;

        int sum = 0;
        sum += getHorizontalDisplacement(homography, lowFr, lowFr);
        sum += getHorizontalDisplacement(homography, lowFr, midFr);
        sum += getHorizontalDisplacement(homography, lowFr, higFr);
        sum += getHorizontalDisplacement(homography, midFr, lowFr);
        sum += getHorizontalDisplacement(homography, midFr, midFr);
        sum += getHorizontalDisplacement(homography, midFr, higFr);
        sum += getHorizontalDisplacement(homography, higFr, lowFr);
        sum += getHorizontalDisplacement(homography, higFr, midFr);
        sum += getHorizontalDisplacement(homography, higFr, higFr);
        return sum / 9;
    }

    /**
     * Compass
     */

    float magneticSumAngle = 0;

    @Override
    public void onSensorChanged(SensorEvent event) {

        float azimuth = Math.round(event.values[0]);
        float pitch = Math.round(event.values[1]);
        float roll = Math.round(event.values[2]);

        //Log.d(TAG, String.valueOf(degreeZ));

        magneticSumAngle = azimuth;

        magneticAngle.setText(String.format("Magnetic angle : %s", String.valueOf(magneticSumAngle)));
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    public void resetImageAngle(View view) {
        refCameraFrame = null;
        sumAngle = magneticSumAngle;
    }
}
