#include <stdio.h>

void convolve(double *inputMatrix, int d1_inputMatrix, int d2_inputMatrix, int h_inputMatrix, int w_inputMatrix, double *kernel, int d1_kernel, int d2_kernel, int h_kernel, int w_kernel, int stride) {

    double result[d1_kernel][d2_kernel][d2_inputMatrix][h_inputMatrix-h_kernel+1][w_inputMatrix-w_kernel+1];
    for (size_t f=0; f<d1_kernel; f++) {
      for (size_t h=0; h<d2_kernel; h++) {
        for (size_t g=0; g<d2_inputMatrix; g++) {
          for (size_t i=0; i<h_inputMatrix-h_kernel+1; i+=stride) {
            for (size_t j=0; j<w_inputMatrix-w_kernel+1; j+=stride) {

              double sum_entirewindow = 0;
              for (size_t k=0; k<h_kernel; k++) {

                double sum_onelineofwindow = 0;
                for (size_t l=0; l<w_kernel; l++) {

                  double sum_alllayersspecificval = 0;
                  for (size_t m=0; m<d1_inputMatrix; m++) {
                    double val = *(inputMatrix + d2_inputMatrix*h_inputMatrix*w_inputMatrix*m + h_inputMatrix*w_inputMatrix*g + w_inputMatrix*(i+k) + (j+l) ) * *(kernel + d2_kernel*h_kernel*w_kernel*f + h_kernel*w_kernel*h + w_kernel*k + l ); //inputMatrix[m][g][i+k][j+l] * kernel[f][h][k][l];
                    sum_alllayersspecificval += val;
                  }
                  sum_onelineofwindow += sum_alllayersspecificval;
                }
                sum_entirewindow += sum_onelineofwindow;
              }
              //printf("%lf\n", sum_entirewindow);
              result[j][h][g][i][j] = sum_entirewindow;
            }
          }
        }
      }
    }

    return result;

}

void test(double *inputMatrix, int d1, int d2, int h, int w, int i, int j, int k, int l) {
  double val = *(inputMatrix + d2*h*w*i + h*w*j + w*k + l);
  printf("%lf\n", val);
}
