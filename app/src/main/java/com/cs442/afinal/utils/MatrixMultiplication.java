package com.cs442.afinal.utils;

/**
 * Created by stymjs0515 on 28/05/2018.
 */

public class MatrixMultiplication {

    public static double[][] multiply(double[][] a, double[][] b) {
        int rowsInA = a.length;
        int columnsInA = a[0].length;
        int columnsInB = b[0].length;
        double[][] c = new double[rowsInA][columnsInB];
        for (int i = 0; i < rowsInA; i++) {
            for (int j = 0; j < columnsInB; j++) {
                for (int k = 0; k < columnsInA; k++) {
                    c[i][j] = c[i][j] + a[i][k] * b[k][j];
                }
            }
        }
        //System.out.println("Product of A and B is");
        for (int i = 0; i < c.length; i++) {
            for (int j = 0; j < c[0].length; j++) {
                //System.out.print(c[i][j] + " ");
            }
            System.out.println();
        }

        return c;
    }
}
