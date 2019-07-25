#include <iostream>;

using namespace std; // will change to calstar when integrating with firmware

const int s = 3; // state size
const int m = 2; // measurement size
const int minfreq = 10;

double[] H  = {1, 0, 0, 0, 0, 1}; // H and HT are the same thing
double[] nH = {-1, 0, 0, 0, 0, -1};
double dt = 1e-3;
double[] A  = {1, dt, dt*dt/2, 0, 1, dt/2, 0, 0, 0}; // to be edited after a flight event
double[] AT = {1, 0, 0, dt, 1, 0, dt*dt/2, dt/2, 0}; // same


double[] matmul(double[] A, double[] B, int ar, int ac, int bc) {
    double product[ar*bc];
    for (int i = 0; i < ar; i++) {
        for (int j = 0; j < bc; j++) {
            product[i * bc + j] = 0;
            for (int k = 0; k < ac; k++) {
                product[i * bc + j] += A[i * ac + k] * B[k * bc + j]
            }
        }
    }
    return product;
}

double[] matmul_sq(double[] A, double[] B, int size) {
    return matmul(A, B, size, size, size);
}

double[] matadd(double[] A, double[] B, int r, int c) {
    double sum[r * c];
    for (int i = 0; i < r * c; i++) {
        sum[i] = A[i] + B[i];
    }
    return sum;
}

double[] matadd_sq(double[] A, double[] B, int size) {
    return matadd(A, B, size, size);
}

double[] transpose(double[] A, int r, int c) {
    double T[size*size];
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
           T[j * r + i] = A[i * c + j]
        }
    }
    return T;
}

double[] inverse3(double[] A) {
    // only works on 3x3 arrays
    double inverted[9];
    inverted[0] = A[4] * A[8] - A[5] * A[7];
    inverted[1] = A[2] * A[7] - A[1] * A[8];
    inverted[2] = A[1] * A[5] - A[2] * A[4];
    inverted[3] = A[5] * A[6] - A[3] * A[8];
    inverted[4] = A[0] * A[8] - A[2] * A[6];
    inverted[5] = A[2] * A[3] - A[0] * A[5];
    inverted[6] = A[3] * A[7] - A[4] * A[6];
    inverted[7] = A[1] * A[6] - A[0] * A[7];
    inverted[8] = A[0] * A[4] - A[1] * A[3];
    double det = A[0] * inverted[0] + A[1] * inverted[3] + A[2] * inverted[6];
    for (int i = 0; i < 9; i++) {
        inverted[i] /= det;
    }
    return inverted;
}

double[] inverse2(double[] A) {
    // only works on 2x2 arrays
    double inverted[4];
    double det = A[0] * A[3] - A[1] * A[2];
    inverted[0] = A[3] / det;
    inverted[1] = -A[1] / det;
    inverted[2] = -A[2] / det;
    inverted[3] = A[0] / det;
    return inverted;
}

double[] kalman_predict_state(double[] state_prev, double[] A) {
    return matmul(A, state_prev, s, s, 1);
    // x <- Ax
}

double[] kalman_predict_cov(double[] A, double[] P_prev, double[] Q) {
    // P <- APA^T + Q
    return matadd_sq(matmul_sq(A, matmul_sq(P_prev, AT, s), s), Q, s);
}

double[] kalman_gain(double[] P, double[] R) {
    // K <- PH^T (HPH^T + R)^(-1)
    double[] PHT = matmul(P, H, s, s, m);
    return matmul(PHT, inverse2(matadd_sq(matmul(H, PHT, m, s, m), R, m)), s, 2, 2);
}

double[] kalman_update_state(double[] state_prev, double[] measurement, double[] gain) {
    double[] error = matadd(measurement, matmul(nH, state_prev, m, s, 1), m, 1);
    return matadd(state_prev, matmul(gain, error, s, m, 1));
}