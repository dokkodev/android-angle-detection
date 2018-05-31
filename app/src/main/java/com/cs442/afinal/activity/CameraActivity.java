package com.cs442.afinal.activity;

import android.hardware.Camera;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;

import com.cs442.afinal.utils.MatrixMultiplication;
import com.cs442.afinal.R;
import com.cs442.afinal.model.ReferenceImage;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.ORB;
import org.opencv.imgproc.Imgproc;

import java.util.LinkedList;
import java.util.List;

public class CameraActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = CameraActivity.class.getName();

    private static int frameCounter = 0;

    private CameraBridgeViewBase mOpenCvCameraView;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_camera);

        mOpenCvCameraView = findViewById(R.id.CvCameraView);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);
    }

    @Override
    public void onPause()
    {
        super.onPause();
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
        return orbFeatureDetect(inputFrame);
    }


    ReferenceImage referenceImage;
    Mat permaCloned;

    private Mat orbFeatureDetect(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        Mat rgba = inputFrame.rgba();

        // Check Frame Counter
        frameCounter++;
        if (frameCounter < 10) {
            return permaCloned;
        }
        frameCounter = 0;

        ORB orbFeature = ORB.create();

        MatOfKeyPoint keyPoints = new MatOfKeyPoint();
        orbFeature.detect(inputFrame.rgba(), keyPoints);

        Mat descriptors = new Mat();
        orbFeature.compute(inputFrame.rgba(), keyPoints, descriptors);

        ReferenceImage newRefImage = new ReferenceImage();
        newRefImage.setDescriptors(descriptors);
        newRefImage.setKeyPoints(keyPoints);

        // Check if new image is usable
        if (descriptors.rows() <= 0 || descriptors.cols() <= 0) {
            return rgba;
        }

        // Check if previous image exists
        if (referenceImage == null) {
            referenceImage = newRefImage;
            return rgba;
        }

        //DescriptorMatcher matcher = DescriptorMatcher.create();

        DescriptorMatcher matcher = DescriptorMatcher.create(4);
        MatOfDMatch matches = new MatOfDMatch();

        Mat refDescriptors = referenceImage.getDescriptors();
        Log.d(TAG, "Number of ref descriptors rows : " + refDescriptors.rows());

        Mat newDescriptors = newRefImage.getDescriptors();
        Log.d(TAG, "Number of new descriptors rows : " + newDescriptors.rows());

        matcher.match(newDescriptors, refDescriptors, matches);
        Log.d(TAG, "Number of matches rows : " + matches.rows());

        List<DMatch> matchesList = matches.toList();
        LinkedList<DMatch> goodMatches = new LinkedList<>();

        for (int i = 0; i < matches.rows(); i++){
            //Log.d(TAG, String.valueOf(matchesList.get(i).distance));
            if (matchesList.get(i).distance < 50) {
                goodMatches.addLast(matchesList.get(i));
            }
        }

//        Collections.sort(goodMatches, new Comparator<DMatch>() {
//            @Override
//            public int compare(DMatch dMatch1, DMatch dMatch2) {
//                return Math.round(dMatch1.distance - dMatch2.distance);
//            }
//        });
//
//        Log.d(TAG, Arrays.toString(goodMatches.subList(0, 4).toArray()));

        Log.d(TAG, "Number of good matches rows : " + goodMatches.size());


        MatOfDMatch gm = new MatOfDMatch();
        gm.fromList(goodMatches);


        LinkedList<Point> refPoints = new LinkedList<>();
        LinkedList<Point> newPoints = new LinkedList<>();
        List<KeyPoint> refKeyPoints = referenceImage.getKeyPoints().toList();
        List<KeyPoint> newKeyPoints = newRefImage.getKeyPoints().toList();

        List<KeyPoint> tmpKeyPoints = new LinkedList<>();

        for (int i = 0; i < goodMatches.size(); i++) {
//            Log.d(TAG,
//                    String.valueOf(goodMatches.get(i).trainIdx) + " " +
//                            String.valueOf(goodMatches.get(i).queryIdx));
            refPoints.addLast(refKeyPoints.get(goodMatches.get(i).trainIdx).pt);
            newPoints.addLast(newKeyPoints.get(goodMatches.get(i).queryIdx).pt);
            tmpKeyPoints.add(newKeyPoints.get(goodMatches.get(i).queryIdx));
        }

        MatOfPoint2f refMatPoints = new MatOfPoint2f();
        MatOfPoint2f newMatPoints = new MatOfPoint2f();
        refMatPoints.fromList(refPoints);
        newMatPoints.fromList(newPoints);

        Mat hg = Calib3d.findHomography(refMatPoints, newMatPoints, Calib3d.FM_RANSAC, 3);
        Log.d(TAG, hg.dump());

        double[][] homography = new double[][]
                {
                        {get(hg, 0, 0), get(hg, 0, 1), get(hg, 0, 2)},
                        {get(hg, 1, 0), get(hg, 1, 1), get(hg, 1, 2)},
                        {get(hg, 2, 0), get(hg, 2, 1), get(hg, 2, 2)}
                };

        double[][] testPoint1 = new double[][]
                {
                        {10},
                        {10},
                        {1}
                };

        MatrixMultiplication.multiply(homography, new double[][] {{0}, {0}, {1}});
        MatrixMultiplication.multiply(homography, new double[][] {{100}, {100}, {1}});
        MatrixMultiplication.multiply(homography, new double[][] {{200}, {200}, {1}});
        MatrixMultiplication.multiply(homography, new double[][] {{0}, {200}, {1}});
        MatrixMultiplication.multiply(homography, new double[][] {{200}, {0}, {1}});

        Mat intrinsics = Mat.eye(3, 3, CvType.CV_32F); // dummy camera matrix
        intrinsics.put(0, 0, 400);
        intrinsics.put(1, 1, 400);
        intrinsics.put(0, 2, 640 / 2);
        intrinsics.put(1, 2, 480 / 2);


        Scalar redColor = new Scalar(255, 0, 0);
        Mat clonedRgba = rgba.clone();
        Imgproc.cvtColor(rgba, clonedRgba, Imgproc.COLOR_RGBA2RGB, 4);

        MatOfKeyPoint temp = new MatOfKeyPoint();
        temp.fromList(tmpKeyPoints);
        Features2d.drawKeypoints(clonedRgba, temp, clonedRgba, redColor, 1);

        // Puts the new image to reference and return
        referenceImage = newRefImage;

        permaCloned = clonedRgba;
        return clonedRgba;
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


                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

//    private boolean safeCameraOpen(int id) {
//        boolean qOpened = false;
//
//        try {
//            releaseCameraAndPreview();
//            mCamera = Camera.open(id);
//            qOpened = (mCamera != null);
//        } catch (Exception e) {
//            Log.e(getString(R.string.app_name), "failed to open Camera");
//            e.printStackTrace();
//        }
//
//        return qOpened;
//    }
//
//    private void releaseCameraAndPreview() {
//        mPreview.setCamera(null);
//        if (mCamera != null) {
//            mCamera.release();
//            mCamera = null;
//        }
//    }
}
