//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: mtimes.cpp
//
// MATLAB Coder version            : 5.6
// C/C++ source code generated on  : 23-Nov-2023 17:42:08
//

// Include Files
#include "mtimes.h"

// Function Definitions
//
// Arguments    : const double A_data[]
//                const int A_size[2]
//                const double B_data[]
//                const int B_size[2]
//                double C_data[]
//                int C_size[2]
// Return Type  : void
//
namespace coder {
namespace internal {
namespace blas {
void mtimes(const double A_data[], const int A_size[2], const double B_data[],
            const int B_size[2], double C_data[], int C_size[2])
{
  int m;
  int n;
  m = A_size[0];
  n = B_size[1];
  C_size[0] = A_size[0];
  C_size[1] = B_size[1];
  for (int j{0}; j < n; j++) {
    int boffset;
    int coffset;
    coffset = j * m;
    boffset = j * 6;
    for (int i{0}; i < m; i++) {
      double s;
      s = 0.0;
      for (int k{0}; k < 6; k++) {
        s += A_data[k * A_size[0] + i] * B_data[boffset + k];
      }
      C_data[coffset + i] = s;
    }
  }
}

} // namespace blas
} // namespace internal
} // namespace coder

//
// File trailer for mtimes.cpp
//
// [EOF]
//
