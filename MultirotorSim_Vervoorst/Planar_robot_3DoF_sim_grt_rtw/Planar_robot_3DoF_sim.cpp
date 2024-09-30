/*
 * Planar_robot_3DoF_sim.cpp
 *
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * Code generation for model "Planar_robot_3DoF_sim".
 *
 * Model version              : 12.40
 * Simulink Coder version : 9.9 (R2023a) 19-Nov-2022
 * C++ source code generated on : Thu Nov 23 02:24:17 2023
 *
 * Target selection: grt.tlc
 * Note: GRT includes extra infrastructure and instrumentation for prototyping
 * Embedded hardware selection: Intel->x86-64 (Windows64)
 * Code generation objective: Debugging
 * Validation result: Not run
 */

#include "Planar_robot_3DoF_sim.h"
#include "rtwtypes.h"
#include <cmath>
#include <cstring>
#include <emmintrin.h>
#include "Planar_robot_3DoF_sim_private.h"

extern "C"
{

#include "rt_nonfinite.h"

}

/* Function for MATLAB Function: '<S1>/calcSysMatrices' */
void Planar_robot_3DoF_sim::Planar_robot_3DoF_sim_eye(real_T b_I[9])
{
  std::memset(&b_I[0], 0, 9U * sizeof(real_T));
  b_I[0] = 1.0;
  b_I[4] = 1.0;
  b_I[8] = 1.0;
}

/* Function for MATLAB Function: '<S1>/calcSysMatrices' */
void Planar_robot_3DoF_sim::Planar_robot_3DoF_sim_diag(const real_T v[3], real_T
  d[9])
{
  std::memset(&d[0], 0, 9U * sizeof(real_T));
  d[0] = v[0];
  d[4] = v[1];
  d[8] = v[2];
}

/* Function for MATLAB Function: '<S1>/calcSysMatrices' */
void Planar_robot_3DoF_sim::Planar_robot_3DoF_sim_repmat(real_T b[18])
{
  static const int8_T a[6]{ 0, 0, 0, 0, 0, 1 };

  for (int32_T itilerow{0}; itilerow < 3; itilerow++) {
    int32_T ibcol;
    ibcol = itilerow * 6;
    for (int32_T k{0}; k < 6; k++) {
      b[ibcol + k] = a[k];
    }
  }
}

/* Function for MATLAB Function: '<S1>/calcSysMatrices' */
boolean_T Planar_robot_3DoF_sim::Planar_robot_3DoF_sim_all(const boolean_T x[3])
{
  int32_T k;
  boolean_T exitg1;
  boolean_T y;
  y = true;
  k = 0;
  exitg1 = false;
  while ((!exitg1) && (k < 3)) {
    if (!x[k]) {
      y = false;
      exitg1 = true;
    } else {
      k++;
    }
  }

  return y;
}

/* Function for MATLAB Function: '<S1>/calcSysMatrices' */
void Planar_robot_3DoF_sim::Planar_robot_3DoF_sim_mtimes(const real_T A_data[],
  const int32_T A_size[2], const real_T B[16], real_T C_data[], int32_T C_size[2])
{
  int32_T m;
  m = A_size[0];
  C_size[0] = A_size[0];
  C_size[1] = 4;
  for (int32_T j{0}; j < 4; j++) {
    int32_T boffset;
    int32_T coffset;
    int32_T scalarLB;
    int32_T vectorUB;
    coffset = j * m;
    boffset = j << 2;
    scalarLB = (m / 2) << 1;
    vectorUB = scalarLB - 2;
    for (int32_T i{0}; i <= vectorUB; i += 2) {
      __m128d tmp;
      __m128d tmp_0;
      tmp = _mm_loadu_pd(&A_data[i]);
      tmp = _mm_mul_pd(tmp, _mm_set1_pd(B[boffset]));
      tmp_0 = _mm_loadu_pd(&A_data[A_size[0] + i]);
      tmp_0 = _mm_mul_pd(tmp_0, _mm_set1_pd(B[boffset + 1]));
      tmp = _mm_add_pd(tmp_0, tmp);
      tmp_0 = _mm_loadu_pd(&A_data[(A_size[0] << 1) + i]);
      tmp_0 = _mm_mul_pd(tmp_0, _mm_set1_pd(B[boffset + 2]));
      tmp = _mm_add_pd(tmp_0, tmp);
      tmp_0 = _mm_loadu_pd(&A_data[3 * A_size[0] + i]);
      tmp_0 = _mm_mul_pd(tmp_0, _mm_set1_pd(B[boffset + 3]));
      tmp = _mm_add_pd(tmp_0, tmp);
      _mm_storeu_pd(&C_data[coffset + i], tmp);
    }

    for (int32_T i{scalarLB}; i < m; i++) {
      real_T s;
      s = A_data[i] * B[boffset];
      s += A_data[A_size[0] + i] * B[boffset + 1];
      s += A_data[(A_size[0] << 1) + i] * B[boffset + 2];
      s += A_data[3 * A_size[0] + i] * B[boffset + 3];
      C_data[coffset + i] = s;
    }
  }
}

/* Function for MATLAB Function: '<S1>/calcSysMatrices' */
void Planar_robot_3DoF_sim::Planar_robot_3DoF_sim_mtimes_p(const real_T A[36],
  const real_T B_data[], const int32_T B_size[2], real_T C_data[], int32_T
  C_size[2])
{
  int32_T b;
  C_size[0] = 6;
  C_size[1] = B_size[1];
  b = B_size[1];
  for (int32_T j{0}; j < b; j++) {
    int32_T coffset;
    coffset = j * 6;
    for (int32_T i{0}; i < 6; i++) {
      real_T s;
      s = 0.0;
      for (int32_T k{0}; k < 6; k++) {
        s += A[k * 6 + i] * B_data[coffset + k];
      }

      C_data[coffset + i] = s;
    }
  }
}

void Planar_robot_3DoF_sim::Planar_rob_binary_expand_op_ccy(real_T in1[18],
  int32_T in2, const real_T in3[36], const real_T in4[18], const real_T in5[9],
  const real_T in6[6], const real_T in7[18])
{
  __m128d tmp;
  __m128d tmp_0;
  real_T in5_0[36];
  real_T in4_data[12];
  real_T tmp_data[12];
  real_T tmp_data_0[12];
  real_T in5_1;
  int32_T in4_size[2];
  int32_T tmp_size[2];
  int32_T tmp_size_0[2];
  int32_T aux_0_1;
  int32_T aux_1_1;
  int32_T i;
  int32_T in5_tmp;
  int32_T loop_ub;
  int32_T stride_0_1;
  int32_T stride_1_1;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* MATLAB Function: '<S1>/calcSysMatrices' */
  in4_size[0] = 6;
  in4_size[1] = in2;
  for (i = 0; i < in2; i++) {
    for (in5_tmp = 0; in5_tmp < 6; in5_tmp++) {
      in4_data[in5_tmp + 6 * i] = in4[6 * i + in5_tmp];
    }
  }

  Planar_robot_3DoF_sim_mtimes_p(in3, in4_data, in4_size, tmp_data, tmp_size);
  in5_0[18] = 0.0;
  in5_0[24] = -in6[2];
  in5_0[30] = in6[1];
  in5_0[19] = in6[2];
  in5_0[25] = 0.0;
  in5_0[31] = -in6[0];
  in5_0[20] = -in6[1];
  in5_0[26] = in6[0];
  in5_0[32] = 0.0;
  for (i = 0; i < 3; i++) {
    in5_1 = in5[3 * i];
    in5_0[6 * i] = in5_1;
    in5_0[6 * i + 3] = 0.0;
    in5_tmp = (i + 3) * 6;
    in5_0[in5_tmp + 3] = in5_1;
    in5_1 = in5[3 * i + 1];
    in5_0[6 * i + 1] = in5_1;
    in5_0[6 * i + 4] = 0.0;
    in5_0[in5_tmp + 4] = in5_1;
    in5_1 = in5[3 * i + 2];
    in5_0[6 * i + 2] = in5_1;
    in5_0[6 * i + 5] = 0.0;
    in5_0[in5_tmp + 5] = in5_1;
  }

  in4_size[0] = 6;
  in4_size[1] = in2;
  for (i = 0; i < in2; i++) {
    for (in5_tmp = 0; in5_tmp < 6; in5_tmp++) {
      in4_data[in5_tmp + 6 * i] = in7[6 * i + in5_tmp];
    }
  }

  Planar_robot_3DoF_sim_mtimes_p(in5_0, in4_data, in4_size, tmp_data_0,
    tmp_size_0);
  stride_0_1 = (tmp_size[1] != 1);
  stride_1_1 = (tmp_size_0[1] != 1);
  aux_0_1 = 0;
  aux_1_1 = 0;
  loop_ub = tmp_size_0[1] == 1 ? tmp_size[1] : tmp_size_0[1];
  for (i = 0; i < loop_ub; i++) {
    for (in5_tmp = 0; in5_tmp <= 4; in5_tmp += 2) {
      tmp = _mm_loadu_pd(&tmp_data[6 * aux_0_1 + in5_tmp]);
      tmp_0 = _mm_loadu_pd(&tmp_data_0[6 * aux_1_1 + in5_tmp]);
      tmp = _mm_sub_pd(tmp, tmp_0);
      _mm_storeu_pd(&in1[in5_tmp + 6 * i], tmp);
    }

    aux_1_1 += stride_1_1;
    aux_0_1 += stride_0_1;
  }

  /* End of MATLAB Function: '<S1>/calcSysMatrices' */
  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
}

/* Function for MATLAB Function: '<S1>/calcSysMatrices' */
void Planar_robot_3DoF_sim::Planar_robot_3DoF_sim_mtimes_pn(const real_T A_data[],
  const int32_T A_size[2], const real_T B_data[], const int32_T B_size[2],
  real_T C_data[], int32_T C_size[2])
{
  int32_T m;
  m = A_size[1];
  C_size[0] = A_size[1];
  C_size[1] = 6;
  for (int32_T j{0}; j < 6; j++) {
    int32_T boffset;
    int32_T coffset;
    coffset = j * m;
    boffset = j * B_size[0];
    for (int32_T i{0}; i < m; i++) {
      real_T s;
      int32_T aoffset;
      aoffset = i * 6;
      s = 0.0;
      for (int32_T k{0}; k < 6; k++) {
        s += A_data[aoffset + k] * B_data[boffset + k];
      }

      C_data[coffset + i] = s;
    }
  }
}

/* Function for MATLAB Function: '<S1>/calcSysMatrices' */
void Planar_robot_3DoF_sim::Planar_robot_3DoF_si_mtimes_pnc(const real_T A_data[],
  const int32_T A_size[2], const real_T B_data[], const int32_T B_size[2],
  real_T C_data[], int32_T C_size[2])
{
  int32_T b;
  int32_T m;
  m = A_size[0];
  C_size[0] = A_size[0];
  C_size[1] = B_size[1];
  b = B_size[1];
  for (int32_T j{0}; j < b; j++) {
    int32_T boffset;
    int32_T coffset;
    coffset = j * m;
    boffset = j * 6;
    for (int32_T i{0}; i < m; i++) {
      real_T s;
      s = 0.0;
      for (int32_T k{0}; k < 6; k++) {
        s += A_data[k * A_size[0] + i] * B_data[boffset + k];
      }

      C_data[coffset + i] = s;
    }
  }
}

void Planar_robot_3DoF_sim::Planar_robo_binary_expand_op_cc(real_T in1[9],
  int32_T in2, const real_T in3_data[], const int32_T in3_size[2], const real_T
  in4[18])
{
  real_T in4_data[18];
  real_T in1_data[9];
  real_T tmp_data[9];
  int32_T in4_size[2];
  int32_T tmp_size[2];
  int32_T aux_0_1;
  int32_T aux_1_1;
  int32_T i;
  int32_T i_0;
  int32_T loop_ub;
  int32_T loop_ub_0;
  int32_T stride_0_0_tmp;
  int32_T stride_1_0;
  int32_T stride_1_1;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* MATLAB Function: '<S1>/calcSysMatrices' */
  in4_size[0] = 6;
  in4_size[1] = in2 + 1;
  for (i_0 = 0; i_0 <= in2; i_0++) {
    for (i = 0; i < 6; i++) {
      in4_data[i + 6 * i_0] = in4[6 * i_0 + i];
    }
  }

  Planar_robot_3DoF_si_mtimes_pnc(in3_data, in3_size, in4_data, in4_size,
    tmp_data, tmp_size);
  loop_ub_0 = tmp_size[0] == 1 ? in2 + 1 : tmp_size[0];
  loop_ub = tmp_size[1] == 1 ? in2 + 1 : tmp_size[1];
  stride_0_0_tmp = (in2 + 1 != 1);
  stride_1_0 = (tmp_size[0] != 1);
  stride_1_1 = (tmp_size[1] != 1);
  aux_0_1 = 0;
  aux_1_1 = 0;
  for (i_0 = 0; i_0 < loop_ub; i_0++) {
    for (i = 0; i < loop_ub_0; i++) {
      in1_data[i + loop_ub_0 * i_0] = in1[i * stride_0_0_tmp + 3 * aux_0_1] +
        tmp_data[i * stride_1_0 + tmp_size[0] * aux_1_1];
    }

    aux_1_1 += stride_1_1;
    aux_0_1 += stride_0_0_tmp;
  }

  for (i_0 = 0; i_0 < loop_ub; i_0++) {
    for (i = 0; i < loop_ub_0; i++) {
      in1[i + 3 * i_0] = in1_data[loop_ub_0 * i_0 + i];
    }
  }

  /* End of MATLAB Function: '<S1>/calcSysMatrices' */
  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
}

/* Function for MATLAB Function: '<S1>/calcSysMatrices' */
void Planar_robot_3DoF_sim::Planar_robot_3DoF_s_mtimes_pnc5(const real_T A_data[],
  const int32_T A_size[2], const real_T B[36], real_T C_data[], int32_T C_size[2])
{
  int32_T m;
  m = A_size[1];
  C_size[0] = A_size[1];
  C_size[1] = 6;
  for (int32_T j{0}; j < 6; j++) {
    int32_T boffset;
    int32_T coffset;
    coffset = j * m;
    boffset = j * 6;
    for (int32_T i{0}; i < m; i++) {
      real_T s;
      int32_T aoffset;
      aoffset = i * 6;
      s = 0.0;
      for (int32_T k{0}; k < 6; k++) {
        s += A_data[aoffset + k] * B[boffset + k];
      }

      C_data[coffset + i] = s;
    }
  }
}

void Planar_robot_3DoF_sim::Planar_robot_binary_expand_op_c(real_T in1[9],
  int32_T in2, const real_T in3[18], real_T in4, const real_T in5[9], const
  real_T in6[9], const real_T in7[9], const real_T in8_data[], const int32_T
  in8_size[2], const real_T in9[18])
{
  real_T in4_0[36];
  real_T in3_data[18];
  real_T tmp_data[18];
  real_T in1_data[9];
  real_T tmp_data_0[9];
  real_T tmp_data_1[9];
  int32_T in3_size[2];
  int32_T tmp_size[2];
  int32_T tmp_size_0[2];
  int32_T aux_0_1;
  int32_T aux_1_1;
  int32_T aux_2_1;
  int32_T i;
  int32_T in4_tmp;
  int32_T in4_tmp_0;
  int32_T loop_ub;
  int32_T loop_ub_0;
  int32_T stride_1_0;
  int32_T stride_1_1;
  int32_T stride_2_0;
  int32_T stride_2_1;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* MATLAB Function: '<S1>/calcSysMatrices' */
  in3_size[0] = 6;
  in3_size[1] = in2 + 1;
  for (i = 0; i <= in2; i++) {
    for (in4_tmp = 0; in4_tmp < 6; in4_tmp++) {
      in3_data[in4_tmp + 6 * i] = in3[6 * i + in4_tmp];
    }
  }

  for (i = 0; i < 3; i++) {
    in4_0[6 * i] = in5[3 * i] * in4;
    in4_tmp = (i + 3) * 6;
    in4_0[in4_tmp] = -in6[i];
    in4_0[6 * i + 3] = in6[3 * i];
    in4_0[in4_tmp + 3] = in7[3 * i] - in7[i];
    in4_tmp_0 = 3 * i + 1;
    in4_0[6 * i + 1] = in5[in4_tmp_0] * in4;
    in4_0[in4_tmp + 1] = -in6[i + 3];
    in4_0[6 * i + 4] = in6[in4_tmp_0];
    in4_0[in4_tmp + 4] = in7[in4_tmp_0] - in7[i + 3];
    in4_tmp_0 = 3 * i + 2;
    in4_0[6 * i + 2] = in5[in4_tmp_0] * in4;
    in4_0[in4_tmp + 2] = -in6[i + 6];
    in4_0[6 * i + 5] = in6[in4_tmp_0];
    in4_0[in4_tmp + 5] = in7[in4_tmp_0] - in7[i + 6];
  }

  Planar_robot_3DoF_s_mtimes_pnc5(in3_data, in3_size, in4_0, tmp_data, tmp_size);
  in3_size[0] = 6;
  in3_size[1] = in2 + 1;
  for (i = 0; i <= in2; i++) {
    for (in4_tmp = 0; in4_tmp < 6; in4_tmp++) {
      in3_data[in4_tmp + 6 * i] = in3[6 * i + in4_tmp];
    }
  }

  Planar_robot_3DoF_si_mtimes_pnc(tmp_data, tmp_size, in3_data, in3_size,
    tmp_data_0, tmp_size_0);
  in3_size[0] = 6;
  in3_size[1] = in2 + 1;
  for (i = 0; i <= in2; i++) {
    for (in4_tmp = 0; in4_tmp < 6; in4_tmp++) {
      in3_data[in4_tmp + 6 * i] = in9[6 * i + in4_tmp];
    }
  }

  Planar_robot_3DoF_si_mtimes_pnc(in8_data, in8_size, in3_data, in3_size,
    tmp_data_1, tmp_size);
  loop_ub_0 = tmp_size[0] == 1 ? tmp_size_0[0] == 1 ? in2 + 1 : tmp_size_0[0] :
    tmp_size[0];
  loop_ub = tmp_size[1] == 1 ? tmp_size_0[1] == 1 ? in2 + 1 : tmp_size_0[1] :
    tmp_size[1];
  in4_tmp_0 = (in2 + 1 != 1);
  stride_1_0 = (tmp_size_0[0] != 1);
  stride_1_1 = (tmp_size_0[1] != 1);
  stride_2_0 = (tmp_size[0] != 1);
  stride_2_1 = (tmp_size[1] != 1);
  aux_0_1 = 0;
  aux_1_1 = 0;
  aux_2_1 = 0;
  for (i = 0; i < loop_ub; i++) {
    for (in4_tmp = 0; in4_tmp < loop_ub_0; in4_tmp++) {
      in1_data[in4_tmp + loop_ub_0 * i] = (in1[in4_tmp * in4_tmp_0 + 3 * aux_0_1]
        + tmp_data_0[in4_tmp * stride_1_0 + tmp_size_0[0] * aux_1_1]) +
        tmp_data_1[in4_tmp * stride_2_0 + tmp_size[0] * aux_2_1];
    }

    aux_2_1 += stride_2_1;
    aux_1_1 += stride_1_1;
    aux_0_1 += in4_tmp_0;
  }

  for (i = 0; i < loop_ub; i++) {
    for (in4_tmp = 0; in4_tmp < loop_ub_0; in4_tmp++) {
      in1[in4_tmp + 3 * i] = in1_data[loop_ub_0 * i + in4_tmp];
    }
  }

  /* End of MATLAB Function: '<S1>/calcSysMatrices' */
  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
}

/* Function for MATLAB Function: '<S1>/calcSysMatrices' */
void Planar_robot_3DoF_sim::Planar_robot_3DoF__mtimes_pnc5a(const real_T A_data[],
  const int32_T A_size[2], const real_T B[9], real_T C_data[], int32_T C_size[2])
{
  int32_T m;
  m = A_size[0];
  C_size[0] = A_size[0];
  C_size[1] = 3;
  for (int32_T j{0}; j < 3; j++) {
    int32_T coffset;
    int32_T scalarLB;
    int32_T vectorUB;
    coffset = j * m;
    scalarLB = (m / 2) << 1;
    vectorUB = scalarLB - 2;
    for (int32_T i{0}; i <= vectorUB; i += 2) {
      __m128d tmp;
      __m128d tmp_0;
      tmp = _mm_loadu_pd(&A_data[i]);
      tmp = _mm_mul_pd(tmp, _mm_set1_pd(B[j]));
      tmp_0 = _mm_loadu_pd(&A_data[A_size[0] + i]);
      tmp_0 = _mm_mul_pd(tmp_0, _mm_set1_pd(B[j + 3]));
      tmp = _mm_add_pd(tmp_0, tmp);
      tmp_0 = _mm_loadu_pd(&A_data[(A_size[0] << 1) + i]);
      tmp_0 = _mm_mul_pd(tmp_0, _mm_set1_pd(B[j + 6]));
      tmp = _mm_add_pd(tmp_0, tmp);
      _mm_storeu_pd(&C_data[coffset + i], tmp);
    }

    for (int32_T i{scalarLB}; i < m; i++) {
      real_T s;
      s = A_data[i] * B[j];
      s += A_data[A_size[0] + i] * B[j + 3];
      s += A_data[(A_size[0] << 1) + i] * B[j + 6];
      C_data[coffset + i] = s;
    }
  }
}

/* Function for MATLAB Function: '<S1>/calcSysMatrices' */
void Planar_robot_3DoF_sim::Planar_robot_3DoF_mtimes_pnc5ag(const real_T A_data[],
  const int32_T A_size[2], real_T C_data[], int32_T *C_size)
{
  int32_T b;
  int32_T scalarLB;
  int32_T vectorUB;
  *C_size = A_size[0];
  b = A_size[0];
  scalarLB = (b / 2) << 1;
  vectorUB = scalarLB - 2;
  for (int32_T i{0}; i <= vectorUB; i += 2) {
    __m128d tmp;
    __m128d tmp_0;
    __m128d tmp_1;
    tmp = _mm_loadu_pd(&A_data[i]);
    tmp_1 = _mm_set1_pd(0.0);
    tmp = _mm_mul_pd(tmp, tmp_1);
    tmp_0 = _mm_loadu_pd(&A_data[A_size[0] + i]);
    tmp_0 = _mm_mul_pd(tmp_0, tmp_1);
    tmp = _mm_add_pd(tmp_0, tmp);
    tmp_0 = _mm_loadu_pd(&A_data[(A_size[0] << 1) + i]);
    tmp_1 = _mm_mul_pd(tmp_0, tmp_1);
    tmp = _mm_add_pd(tmp_1, tmp);
    _mm_storeu_pd(&C_data[i], tmp);
  }

  for (int32_T i{scalarLB}; i < b; i++) {
    real_T s;
    s = A_data[i] * 0.0;
    s += A_data[A_size[0] + i] * 0.0;
    s += A_data[(A_size[0] << 1) + i] * 0.0;
    C_data[i] = s;
  }
}

void Planar_robot_3DoF_sim::Planar_robot_3_binary_expand_op(real_T in1[3],
  int32_T in2, const real_T in3_data[], const int32_T in3_size[2], const real_T
  in4[16])
{
  real_T in3_data_0[9];
  real_T in4_0[9];
  real_T tmp_data[9];
  real_T in1_data[3];
  real_T tmp_data_0[3];
  int32_T in3_size_0[2];
  int32_T tmp_size[2];
  int32_T in4_tmp;
  int32_T stride_0_1;
  int32_T stride_1_1;
  int32_T tmp_size_0;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* MATLAB Function: '<S1>/calcSysMatrices' */
  in3_size_0[0] = in2 + 1;
  in3_size_0[1] = 3;
  for (tmp_size_0 = 0; tmp_size_0 < 3; tmp_size_0++) {
    for (in4_tmp = 0; in4_tmp <= in2; in4_tmp++) {
      in3_data_0[in4_tmp + in3_size_0[0] * tmp_size_0] = in3_data[in3_size[0] *
        tmp_size_0 + in4_tmp];
    }

    in4_tmp = tmp_size_0 << 2;
    in4_0[3 * tmp_size_0] = in4[in4_tmp];
    in4_0[3 * tmp_size_0 + 1] = in4[in4_tmp + 1];
    in4_0[3 * tmp_size_0 + 2] = in4[in4_tmp + 2];
  }

  Planar_robot_3DoF__mtimes_pnc5a(in3_data_0, in3_size_0, in4_0, tmp_data,
    tmp_size);
  Planar_robot_3DoF_mtimes_pnc5ag(tmp_data, tmp_size, tmp_data_0, &tmp_size_0);
  in4_tmp = in2 + 1;
  stride_0_1 = (in2 + 1 != 1);
  stride_1_1 = (tmp_size_0 != 1);
  for (tmp_size_0 = 0; tmp_size_0 < in4_tmp; tmp_size_0++) {
    in1_data[tmp_size_0] = in1[tmp_size_0 * stride_0_1] + tmp_data_0[tmp_size_0 *
      stride_1_1];
  }

  if (in4_tmp - 1 >= 0) {
    std::memcpy(&in1[0], &in1_data[0], static_cast<uint32_T>(in4_tmp) * sizeof
                (real_T));
  }

  /* End of MATLAB Function: '<S1>/calcSysMatrices' */
  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
}

void rt_invd3x3_snf(const real_T u[9], real_T y[9])
{
  real_T x[9];
  real_T absx11;
  real_T absx21;
  real_T absx31;
  real_T y_0;
  int32_T p1;
  int32_T six;
  int32_T three;
  int32_T zero;
  std::memcpy(&x[0], &u[0], 9U * sizeof(real_T));
  three = 3;
  six = 6;
  p1 = 0;
  absx31 = x[0];
  absx11 = std::abs(absx31);
  absx31 = x[1];
  absx21 = std::abs(absx31);
  absx31 = x[2];
  absx31 = std::abs(absx31);
  if ((absx21 > absx11) && (absx21 > absx31)) {
    p1 = 3;
    three = 0;
    absx21 = x[0];
    x[0] = x[1];
    x[1] = absx21;
    absx21 = x[3];
    x[3] = x[4];
    x[4] = absx21;
    absx21 = x[6];
    x[6] = x[7];
    x[7] = absx21;
  } else if (absx31 > absx11) {
    p1 = 6;
    six = 0;
    absx21 = x[0];
    x[0] = x[2];
    x[2] = absx21;
    absx21 = x[3];
    x[3] = x[5];
    x[5] = absx21;
    absx21 = x[6];
    x[6] = x[8];
    x[8] = absx21;
  }

  absx31 = x[1];
  y_0 = x[0];
  absx31 /= y_0;
  x[1] = absx31;
  absx31 = x[2];
  y_0 = x[0];
  absx31 /= y_0;
  x[2] = absx31;
  x[4] -= x[1] * x[3];
  x[5] -= x[2] * x[3];
  x[7] -= x[1] * x[6];
  x[8] -= x[2] * x[6];
  absx31 = x[5];
  y_0 = std::abs(absx31);
  absx31 = x[4];
  absx31 = std::abs(absx31);
  if (y_0 > absx31) {
    zero = three;
    three = six;
    six = zero;
    absx21 = x[1];
    x[1] = x[2];
    x[2] = absx21;
    absx21 = x[4];
    x[4] = x[5];
    x[5] = absx21;
    absx21 = x[7];
    x[7] = x[8];
    x[8] = absx21;
  }

  absx31 = x[5];
  y_0 = x[4];
  absx31 /= y_0;
  x[5] = absx31;
  x[8] -= x[5] * x[7];
  absx31 = x[1] * x[5] - x[2];
  y_0 = x[8];
  absx11 = absx31 / y_0;
  absx31 = -(x[7] * absx11 + x[1]);
  y_0 = x[4];
  absx21 = absx31 / y_0;
  zero = p1;
  absx31 = (1.0 - x[3] * absx21) - x[6] * absx11;
  y_0 = x[0];
  absx31 /= y_0;
  y[zero] = absx31;
  zero = p1 + 1;
  y[zero] = absx21;
  zero = p1 + 2;
  y[zero] = absx11;
  absx31 = -x[5];
  y_0 = x[8];
  absx11 = absx31 / y_0;
  absx31 = 1.0 - x[7] * absx11;
  y_0 = x[4];
  absx21 = absx31 / y_0;
  zero = three;
  absx31 = -(x[3] * absx21 + x[6] * absx11);
  y_0 = x[0];
  absx31 /= y_0;
  y[zero] = absx31;
  zero = three + 1;
  y[zero] = absx21;
  zero = three + 2;
  y[zero] = absx11;
  y_0 = x[8];
  absx11 = 1.0 / y_0;
  absx31 = -x[7] * absx11;
  y_0 = x[4];
  absx21 = absx31 / y_0;
  zero = six;
  absx31 = -(x[3] * absx21 + x[6] * absx11);
  y_0 = x[0];
  absx31 /= y_0;
  y[zero] = absx31;
  zero = six + 1;
  y[zero] = absx21;
  zero = six + 2;
  y[zero] = absx11;
}

/* Model step function */
void Planar_robot_3DoF_sim::step()
{
  static const int8_T c_b_0[9]{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };

  __m128d tmp_1;
  __m128d tmp_2;
  real_T ROBOT_Mass[108];
  real_T ROBOT_Mass_data[108];
  real_T Htm_data[48];
  real_T ROBOT_g0[48];
  real_T ROBOT_g0_data[48];
  real_T invAd[36];
  real_T jointOrigins_0[36];
  real_T J[18];
  real_T J_data[18];
  real_T J_pre[18];
  real_T ROBOT_csi[18];
  real_T dJ[18];
  real_T dJ_pre[18];
  real_T tmp_data_2[18];
  real_T O_Htm[16];
  real_T O_Htm_0[16];
  real_T O_Htm_pre[16];
  real_T J_pre_data[12];
  real_T tmp_data[12];
  real_T tmp_data_1[12];
  real_T E_tmp[9];
  real_T S[9];
  real_T b_I[9];
  real_T b_I_0[9];
  real_T b_S[9];
  real_T b_y[9];
  real_T c_S[9];
  real_T c_y[9];
  real_T d_y[9];
  real_T jointOrigins[9];
  real_T tmp[9];
  real_T tmp2[9];
  real_T tmp_0[9];
  real_T y[9];
  real_T inertialTwist[6];
  real_T inertialTwist_pre[6];
  real_T relTwist[6];
  real_T tmp_data_0[3];
  real_T tmp_data_3[3];
  real_T Iz;
  real_T O_Htm_1;
  real_T O_Htm_pre_0;
  real_T ROBOT_Mass_0;
  real_T ROBOT_Mass_1;
  real_T ROBOT_csi_1;
  real_T S_0;
  real_T m;
  real_T tmp2_0;
  int32_T Htm_size[2];
  int32_T J_pre_size[2];
  int32_T J_size[2];
  int32_T J_size_0[2];
  int32_T ROBOT_Mass_size[2];
  int32_T ROBOT_g0_size[2];
  int32_T tmp_size[2];
  int32_T tmp_size_0[2];
  int32_T tmp_size_1[2];
  int32_T tmp_size_2[2];
  int32_T tmp_size_3[2];
  int32_T tmp_size_4[2];
  int32_T tmp_size_5[2];
  int32_T O_Htm_tmp;
  int32_T O_Htm_tmp_0;
  int32_T ROBOT_Mass_tmp;
  int32_T i;
  int32_T idxStart_4Row;
  int32_T idxStart_6Row;
  int8_T c_b;
  boolean_T ROBOT_csi_0[3];

  /* Sin: '<Root>/Sine Wave Function3' */
  Planar_robot_3DoF_sim_B.j1_torque = std::sin(2.0 * (&Planar_robot_3DoF_sim_M
    )->Timing.t[0]);

  /* Sin: '<Root>/Sine Wave Function1' */
  Planar_robot_3DoF_sim_B.j2_torque = std::sin(13.0 * (&Planar_robot_3DoF_sim_M
    )->Timing.t[0]) * 0.0;

  /* Sin: '<Root>/Sine Wave Function2' */
  Planar_robot_3DoF_sim_B.j3_torque = std::sin(4.0 * (&Planar_robot_3DoF_sim_M
    )->Timing.t[0]) * 0.0;

  /* SignalConversion generated from: '<Root>/To Workspace1' */
  Planar_robot_3DoF_sim_B.joint_torque[0] = Planar_robot_3DoF_sim_B.j1_torque;
  Planar_robot_3DoF_sim_B.joint_torque[1] = Planar_robot_3DoF_sim_B.j2_torque;
  Planar_robot_3DoF_sim_B.joint_torque[2] = Planar_robot_3DoF_sim_B.j3_torque;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* MATLAB Function: '<S1>/calcSysMatrices' incorporates:
   *  Constant: '<Root>/arm_length'
   *  Constant: '<Root>/arm_link_radius'
   */
  /* :  ROBOT = Planar3DoF(rho, radius, arm_length); */
  /* :  E = eye(3); */
  Planar_robot_3DoF_sim_eye(E_tmp);

  /* :  o = [0 0 0].'; */
  /* :  wz = [0 0 1].'; */
  /* :  jointOrigins = [ */
  /* :      0, arm_length(1), arm_length(2) */
  /* :      0, 0, 0 */
  /* :      0, 0, 0]; */
  jointOrigins[0] = 0.0;
  jointOrigins[3] = 1.0;
  jointOrigins[6] = 0.5;

  /* :  TCP_T_EE = [E, [arm_length(3) 0 0]'; o.', 1]; */
  /* :  Mass = cell(3,1); */
  /* :  g0 = cell(3,1); */
  /* :  for iLink = 1:3 */
  /* :  m = pi * radius^2 * arm_length(iLink) * rho; */
  /* :  com = [arm_length(iLink)/2, 0, 0]; */
  /* :  Ix = 0.5 * m * radius^2; */
  /* :  Iz = (1/4) * m * radius^2 + (1/12) * m * arm_length(iLink)^2; */
  /* :  body.Mass = m; */
  /* :  body.CenterOfMass = com; */
  /* :  body.Inertia = [Ix, Iz, Iz, 0, 0, 0]; */
  /* :  Mass{iLink} = Inertia(body.Mass, body.CenterOfMass, diag(body.Inertia(1:3))); */
  inertialTwist_pre[0] = 0.00025132741228718348;
  inertialTwist_pre[1] = 0.10484541882580335;
  inertialTwist_pre[2] = 0.10484541882580335;
  inertialTwist_pre[3] = 0.0;
  inertialTwist_pre[4] = 0.0;
  inertialTwist_pre[5] = 0.0;
  Planar_robot_3DoF_sim_diag(&inertialTwist_pre[0], tmp2);

  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */

  /* Constant: '<Root>/init_pos' */
  /* :  S = Skew(c); */
  /* :  out = [ */
  /* :          0,         -vec(3),   vec(2) */
  /* :          vec(3),    0,         -vec(1) */
  /* :          -vec(2),   vec(1),    0 */
  /* :      ]; */
  Planar_robot_3DoF_sim_B.init_pos[0] = 0.0;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* DiscreteIntegrator: '<S1>/Discrete-Time Integrator1' */
  Planar_robot_3DoF_sim_B.q[0] =
    Planar_robot_3DoF_sim_DW.DiscreteTimeIntegrator1_DSTATE[0];

  /* DiscreteIntegrator: '<S1>/Discrete-Time Integrator' */
  Planar_robot_3DoF_sim_B.dq[0] =
    Planar_robot_3DoF_sim_DW.DiscreteTimeIntegrator_DSTATE[0];

  /* MATLAB Function: '<S1>/calcSysMatrices' */
  jointOrigins[1] = 0.0;
  jointOrigins[2] = 0.0;
  S[0] = 0.0;

  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */

  /* Constant: '<Root>/init_pos' */
  Planar_robot_3DoF_sim_B.init_pos[1] = 0.0;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* DiscreteIntegrator: '<S1>/Discrete-Time Integrator1' */
  Planar_robot_3DoF_sim_B.q[1] =
    Planar_robot_3DoF_sim_DW.DiscreteTimeIntegrator1_DSTATE[1];

  /* DiscreteIntegrator: '<S1>/Discrete-Time Integrator' */
  Planar_robot_3DoF_sim_B.dq[1] =
    Planar_robot_3DoF_sim_DW.DiscreteTimeIntegrator_DSTATE[1];

  /* MATLAB Function: '<S1>/calcSysMatrices' */
  jointOrigins[4] = 0.0;
  jointOrigins[5] = 0.0;
  S[3] = 0.0;

  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */

  /* Constant: '<Root>/init_pos' */
  Planar_robot_3DoF_sim_B.init_pos[2] = 0.0;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* DiscreteIntegrator: '<S1>/Discrete-Time Integrator1' */
  Planar_robot_3DoF_sim_B.q[2] =
    Planar_robot_3DoF_sim_DW.DiscreteTimeIntegrator1_DSTATE[2];

  /* DiscreteIntegrator: '<S1>/Discrete-Time Integrator' */
  Planar_robot_3DoF_sim_B.dq[2] =
    Planar_robot_3DoF_sim_DW.DiscreteTimeIntegrator_DSTATE[2];

  /* MATLAB Function: '<S1>/calcSysMatrices' incorporates:
   *  Constant: '<Root>/arm_length'
   *  Constant: '<Root>/arm_link_radius'
   */
  jointOrigins[7] = 0.0;
  jointOrigins[8] = 0.0;
  S[6] = 0.0;
  S[1] = 0.0;
  S[4] = 0.0;
  S[7] = -0.5;
  S[2] = -0.0;
  S[5] = 0.5;
  S[8] = 0.0;

  /* :  out = [ */
  /* :          m*eye(3),   -m*S */
  /* :          m*S,        I - m*S*S */
  /* :      ]; */
  /* :  g0{iLink} = [E, jointOrigins(:,iLink); o.', 1]; */
  /* :  m = pi * radius^2 * arm_length(iLink) * rho; */
  /* :  com = [arm_length(iLink)/2, 0, 0]; */
  /* :  Ix = 0.5 * m * radius^2; */
  /* :  Iz = (1/4) * m * radius^2 + (1/12) * m * arm_length(iLink)^2; */
  /* :  body.Mass = m; */
  /* :  body.CenterOfMass = com; */
  /* :  body.Inertia = [Ix, Iz, Iz, 0, 0, 0]; */
  /* :  Mass{iLink} = Inertia(body.Mass, body.CenterOfMass, diag(body.Inertia(1:3))); */
  inertialTwist_pre[0] = 0.00012566370614359174;
  inertialTwist_pre[1] = 0.013152801243029267;
  inertialTwist_pre[2] = 0.013152801243029267;
  inertialTwist_pre[3] = 0.0;
  inertialTwist_pre[4] = 0.0;
  inertialTwist_pre[5] = 0.0;
  Planar_robot_3DoF_sim_diag(&inertialTwist_pre[0], b_I);

  /* :  S = Skew(c); */
  /* :  out = [ */
  /* :          0,         -vec(3),   vec(2) */
  /* :          vec(3),    0,         -vec(1) */
  /* :          -vec(2),   vec(1),    0 */
  /* :      ]; */
  b_S[0] = 0.0;
  b_S[3] = 0.0;
  b_S[6] = 0.0;
  b_S[1] = 0.0;
  b_S[4] = 0.0;
  b_S[7] = -0.25;
  b_S[2] = -0.0;
  b_S[5] = 0.25;
  b_S[8] = 0.0;

  /* :  out = [ */
  /* :          m*eye(3),   -m*S */
  /* :          m*S,        I - m*S*S */
  /* :      ]; */
  for (i = 0; i < 9; i++) {
    c_b = c_b_0[i];
    y[i] = 1.2566370614359172 * static_cast<real_T>(c_b);
    b_y[i] = 1.2566370614359172 * S[i];
    c_y[i] = 0.62831853071795862 * static_cast<real_T>(c_b);
    d_y[i] = 0.62831853071795862 * b_S[i];
  }

  /* :  g0{iLink} = [E, jointOrigins(:,iLink); o.', 1]; */
  /* :  m = pi * radius^2 * arm_length(iLink) * rho; */
  /* :  com = [arm_length(iLink)/2, 0, 0]; */
  /* :  Ix = 0.5 * m * radius^2; */
  /* :  Iz = (1/4) * m * radius^2 + (1/12) * m * arm_length(iLink)^2; */
  /* :  body.Mass = m; */
  /* :  body.CenterOfMass = com; */
  /* :  body.Inertia = [Ix, Iz, Iz, 0, 0, 0]; */
  /* :  Mass{iLink} = Inertia(body.Mass, body.CenterOfMass, diag(body.Inertia(1:3))); */
  /* :  S = Skew(c); */
  /* :  out = [ */
  /* :          0,         -vec(3),   vec(2) */
  /* :          vec(3),    0,         -vec(1) */
  /* :          -vec(2),   vec(1),    0 */
  /* :      ]; */
  c_S[1] = 0.0;
  c_S[4] = 0.0;
  c_S[7] = -0.05;
  c_S[2] = -0.0;
  c_S[5] = 0.05;
  c_S[8] = 0.0;

  /* :  out = [ */
  /* :          m*eye(3),   -m*S */
  /* :          m*S,        I - m*S*S */
  /* :      ]; */
  /* :  g0{iLink} = [E, jointOrigins(:,iLink); o.', 1]; */
  /* :  ROBOT.csi = repmat([o; wz], 3, 1); */
  Planar_robot_3DoF_sim_repmat(ROBOT_csi);

  /* :  ROBOT.g_vec = [0 0 0]'; */
  /* :  ROBOT.g0 = [g0{1}; g0{2}; g0{3}]; */
  for (i = 0; i < 3; i++) {
    c_S[3 * i] = 0.0;
    ROBOT_g0[12 * i] = E_tmp[3 * i];
    ROBOT_g0[12 * i + 1] = E_tmp[3 * i + 1];
    ROBOT_g0[12 * i + 2] = E_tmp[3 * i + 2];
    ROBOT_g0[i + 36] = 0.0;
  }

  ROBOT_g0[3] = 0.0;
  ROBOT_g0[15] = 0.0;
  ROBOT_g0[27] = 0.0;
  ROBOT_g0[39] = 1.0;
  for (i = 0; i < 3; i++) {
    ROBOT_g0[12 * i + 4] = E_tmp[3 * i];
    ROBOT_g0[12 * i + 5] = E_tmp[3 * i + 1];
    ROBOT_g0[12 * i + 6] = E_tmp[3 * i + 2];
    ROBOT_g0[i + 40] = jointOrigins[i + 3];
  }

  ROBOT_g0[7] = 0.0;
  ROBOT_g0[19] = 0.0;
  ROBOT_g0[31] = 0.0;
  ROBOT_g0[43] = 1.0;
  for (i = 0; i < 3; i++) {
    ROBOT_g0[12 * i + 8] = E_tmp[3 * i];
    ROBOT_g0[12 * i + 9] = E_tmp[3 * i + 1];
    ROBOT_g0[12 * i + 10] = E_tmp[3 * i + 2];
    ROBOT_g0[i + 44] = jointOrigins[i + 6];
  }

  ROBOT_g0[11] = 0.0;
  ROBOT_g0[23] = 0.0;
  ROBOT_g0[35] = 0.0;
  ROBOT_g0[47] = 1.0;

  /* :  ROBOT.Mass = [Mass{1}; Mass{2}; Mass{3}]; */
  inertialTwist_pre[0] = 2.513274122871835E-5;
  inertialTwist_pre[1] = 0.00011728612573401898;
  inertialTwist_pre[2] = 0.00011728612573401898;
  inertialTwist_pre[3] = 0.0;
  inertialTwist_pre[4] = 0.0;
  inertialTwist_pre[5] = 0.0;
  Planar_robot_3DoF_sim_diag(&inertialTwist_pre[0], tmp);
  for (i = 0; i < 3; i++) {
    for (O_Htm_tmp = 0; O_Htm_tmp < 3; O_Htm_tmp++) {
      ROBOT_csi_1 = 1.2566370614359172 * S[i] * S[3 * O_Htm_tmp];
      Iz = 0.62831853071795862 * b_S[i] * b_S[3 * O_Htm_tmp];
      m = 0.12566370614359174 * c_S[i] * c_S[3 * O_Htm_tmp];
      ROBOT_Mass_tmp = 3 * O_Htm_tmp + 1;
      ROBOT_csi_1 += S[i + 3] * 1.2566370614359172 * S[ROBOT_Mass_tmp];
      Iz += b_S[i + 3] * 0.62831853071795862 * b_S[ROBOT_Mass_tmp];
      m += c_S[i + 3] * 0.12566370614359174 * c_S[ROBOT_Mass_tmp];
      ROBOT_Mass_tmp = 3 * O_Htm_tmp + 2;
      ROBOT_csi_1 += S[i + 6] * 1.2566370614359172 * S[ROBOT_Mass_tmp];
      Iz += b_S[i + 6] * 0.62831853071795862 * b_S[ROBOT_Mass_tmp];
      m += c_S[i + 6] * 0.12566370614359174 * c_S[ROBOT_Mass_tmp];
      idxStart_4Row = 3 * O_Htm_tmp + i;
      jointOrigins[idxStart_4Row] = tmp2[idxStart_4Row] - ROBOT_csi_1;
      b_I_0[idxStart_4Row] = b_I[idxStart_4Row] - Iz;
      tmp_0[idxStart_4Row] = tmp[idxStart_4Row] - m;
      idxStart_4Row = 3 * i + O_Htm_tmp;
      ROBOT_Mass_tmp = 18 * i + O_Htm_tmp;
      ROBOT_Mass[ROBOT_Mass_tmp] = y[idxStart_4Row];
      ROBOT_Mass[O_Htm_tmp + 18 * (i + 3)] = S[idxStart_4Row] *
        -1.2566370614359172;
      ROBOT_Mass[ROBOT_Mass_tmp + 3] = b_y[idxStart_4Row];
    }
  }

  for (i = 0; i < 3; i++) {
    idxStart_4Row = (i + 3) * 18;
    ROBOT_Mass[idxStart_4Row + 3] = jointOrigins[3 * i];
    ROBOT_Mass[18 * i + 6] = c_y[3 * i];
    ROBOT_Mass[idxStart_4Row + 6] = b_S[3 * i] * -0.62831853071795862;
    ROBOT_Mass[18 * i + 9] = d_y[3 * i];
    ROBOT_Mass[idxStart_4Row + 9] = b_I_0[3 * i];
    ROBOT_Mass[18 * i + 12] = static_cast<real_T>(c_b_0[3 * i]) *
      0.12566370614359174;
    Iz = c_S[3 * i];
    ROBOT_Mass[idxStart_4Row + 12] = -0.12566370614359174 * Iz;
    ROBOT_Mass[18 * i + 15] = 0.12566370614359174 * Iz;
    ROBOT_Mass[idxStart_4Row + 15] = tmp_0[3 * i];
    ROBOT_Mass_tmp = 3 * i + 1;
    ROBOT_Mass[idxStart_4Row + 4] = jointOrigins[ROBOT_Mass_tmp];
    ROBOT_Mass[18 * i + 7] = c_y[ROBOT_Mass_tmp];
    ROBOT_Mass[idxStart_4Row + 7] = b_S[ROBOT_Mass_tmp] * -0.62831853071795862;
    ROBOT_Mass[18 * i + 10] = d_y[ROBOT_Mass_tmp];
    ROBOT_Mass[idxStart_4Row + 10] = b_I_0[ROBOT_Mass_tmp];
    ROBOT_Mass[18 * i + 13] = static_cast<real_T>(c_b_0[ROBOT_Mass_tmp]) *
      0.12566370614359174;
    Iz = c_S[ROBOT_Mass_tmp];
    ROBOT_Mass[idxStart_4Row + 13] = -0.12566370614359174 * Iz;
    ROBOT_Mass[18 * i + 16] = 0.12566370614359174 * Iz;
    ROBOT_Mass[idxStart_4Row + 16] = tmp_0[ROBOT_Mass_tmp];
    ROBOT_Mass_tmp = 3 * i + 2;
    ROBOT_Mass[idxStart_4Row + 5] = jointOrigins[ROBOT_Mass_tmp];
    ROBOT_Mass[18 * i + 8] = c_y[ROBOT_Mass_tmp];
    ROBOT_Mass[idxStart_4Row + 8] = b_S[ROBOT_Mass_tmp] * -0.62831853071795862;
    ROBOT_Mass[18 * i + 11] = d_y[ROBOT_Mass_tmp];
    ROBOT_Mass[idxStart_4Row + 11] = b_I_0[ROBOT_Mass_tmp];
    ROBOT_Mass[18 * i + 14] = static_cast<real_T>(c_b_0[ROBOT_Mass_tmp]) *
      0.12566370614359174;
    Iz = c_S[ROBOT_Mass_tmp];
    ROBOT_Mass[idxStart_4Row + 14] = -0.12566370614359174 * Iz;
    ROBOT_Mass[18 * i + 17] = 0.12566370614359174 * Iz;
    ROBOT_Mass[idxStart_4Row + 17] = tmp_0[ROBOT_Mass_tmp];
  }

  /* :  ROBOT.tcp_t_ee = TCP_T_EE; */
  /* :  ndof = size(ROBOT.g0, 1)/4; */
  /* :  q = reshape(q, ndof, 1); */
  /* :  dq = reshape(dq, ndof, 1); */
  /* :  M = zeros(ndof, ndof); */
  /* :  dM = zeros(ndof, ndof); */
  /* :  CC = zeros(ndof, ndof); */
  std::memset(&Planar_robot_3DoF_sim_B.M[0], 0, 9U * sizeof(real_T));
  std::memset(&Planar_robot_3DoF_sim_B.CC[0], 0, 9U * sizeof(real_T));

  /* :  g = zeros(ndof, 1); */
  Planar_robot_3DoF_sim_B.g[0] = 0.0;
  Planar_robot_3DoF_sim_B.g[1] = 0.0;
  Planar_robot_3DoF_sim_B.g[2] = 0.0;

  /* :  J_pre = zeros(6, ndof); */
  /* :  dJ_pre = zeros(6, ndof); */
  std::memset(&J_pre[0], 0, 18U * sizeof(real_T));
  std::memset(&dJ_pre[0], 0, 18U * sizeof(real_T));

  /* :  inertialTwist_pre = zeros(6,1); */
  for (i = 0; i < 6; i++) {
    inertialTwist_pre[i] = 0.0;
  }

  /* :  O_Htm_pre = eye(4); */
  std::memset(&O_Htm_pre[0], 0, sizeof(real_T) << 4U);
  O_Htm_pre[0] = 1.0;
  O_Htm_pre[5] = 1.0;
  O_Htm_pre[10] = 1.0;
  O_Htm_pre[15] = 1.0;

  /* :  for iLink = 1:ndof */
  tmp[0] = 0.0;
  tmp[4] = 0.0;
  tmp[8] = 0.0;
  for (ROBOT_Mass_tmp = 0; ROBOT_Mass_tmp < 3; ROBOT_Mass_tmp++) {
    /* :  idxStart_4Row = 4*(iLink-1) + 1; */
    idxStart_4Row = ROBOT_Mass_tmp << 2;

    /* :  idxEnd_4Row = idxStart_4Row + 3; */
    /* :  idxStart_6Row = 6*(iLink-1) + 1; */
    idxStart_6Row = 6 * ROBOT_Mass_tmp;

    /* :  idxEnd_6Row = idxStart_6Row + 5; */
    /* :  relBodyJac = ROBOT.csi(idxStart_6Row:idxEnd_6Row); */
    /* :  relBodyJac = reshape(relBodyJac, 6, 1); */
    /* :  relTwist = relBodyJac * dq(iLink); */
    ROBOT_csi_1 = Planar_robot_3DoF_sim_B.dq[ROBOT_Mass_tmp];
    for (i = 0; i <= 4; i += 2) {
      tmp_2 = _mm_loadu_pd(&ROBOT_csi[idxStart_6Row + i]);
      tmp_2 = _mm_mul_pd(tmp_2, _mm_set1_pd(ROBOT_csi_1));
      _mm_storeu_pd(&relTwist[i], tmp_2);
    }

    /* :  Htm = ROBOT.g0(idxStart_4Row:idxEnd_4Row, 1:4); */
    /* :  Htm = Htm * Exponential(relBodyJac, q(iLink)); */
    /* :  o = [0 0 0].'; */
    /* :  I = eye( 3 ); */
    /* :  v = csi(1:3); */
    /* :  w = csi(4:6); */
    /* :  if all( w == o ) */
    Iz = ROBOT_csi[idxStart_6Row + 3];
    ROBOT_csi_0[0] = (Iz == 0.0);
    ROBOT_csi_1 = ROBOT_csi[idxStart_6Row + 4];
    ROBOT_csi_0[1] = (ROBOT_csi_1 == 0.0);
    m = ROBOT_csi[idxStart_6Row + 5];
    ROBOT_csi_0[2] = (m == 0.0);
    if (Planar_robot_3DoF_sim_all(ROBOT_csi_0)) {
      /* :  out = [I v*theta; zeros(1,3) 1]; */
      ROBOT_csi_1 = Planar_robot_3DoF_sim_B.q[ROBOT_Mass_tmp];
      for (i = 0; i < 3; i++) {
        O_Htm_tmp_0 = i << 2;
        O_Htm[O_Htm_tmp_0] = E_tmp[3 * i];
        O_Htm[O_Htm_tmp_0 + 1] = E_tmp[3 * i + 1];
        O_Htm[O_Htm_tmp_0 + 2] = E_tmp[3 * i + 2];
        O_Htm[i + 12] = ROBOT_csi[idxStart_6Row + i] * ROBOT_csi_1;
      }

      O_Htm[3] = 0.0;
      O_Htm[7] = 0.0;
      O_Htm[11] = 0.0;
      O_Htm[15] = 1.0;
    } else {
      /* :  else */
      /* :  w_hat = Skew( w ); */
      /* :  out = [ */
      /* :      0         -vec(3)   vec(2) */
      /* :      vec(3)    0         -vec(1) */
      /* :      -vec(2)   vec(1)    0 */
      /* :      ]; */
      jointOrigins[0] = 0.0;
      jointOrigins[3] = -m;
      jointOrigins[6] = ROBOT_csi_1;
      jointOrigins[1] = m;
      jointOrigins[4] = 0.0;
      jointOrigins[7] = -Iz;
      jointOrigins[2] = -ROBOT_csi_1;
      jointOrigins[5] = Iz;
      jointOrigins[8] = 0.0;

      /* :  R_minus_I = w_hat * sin( theta ) + w_hat * w_hat * ( 1 - cos(theta) ); */
      m = std::sin(Planar_robot_3DoF_sim_B.q[ROBOT_Mass_tmp]);
      Iz = std::cos(Planar_robot_3DoF_sim_B.q[ROBOT_Mass_tmp]);

      /* :  out = [I+R_minus_I zeros(3,1); zeros(1,3) 1]; */
      ROBOT_csi_1 = 1.0 - Iz;
      for (i = 0; i < 3; i++) {
        for (O_Htm_tmp = 0; O_Htm_tmp < 3; O_Htm_tmp++) {
          Iz = jointOrigins[3 * O_Htm_tmp] * jointOrigins[i];
          Iz += jointOrigins[3 * O_Htm_tmp + 1] * jointOrigins[i + 3];
          Iz += jointOrigins[3 * O_Htm_tmp + 2] * jointOrigins[i + 6];
          tmp2[i + 3 * O_Htm_tmp] = Iz;
        }
      }

      for (i = 0; i < 3; i++) {
        O_Htm_tmp_0 = i << 2;
        O_Htm[O_Htm_tmp_0] = (jointOrigins[3 * i] * m + tmp2[3 * i] *
                              ROBOT_csi_1) + E_tmp[3 * i];
        O_Htm_tmp = 3 * i + 1;
        O_Htm[O_Htm_tmp_0 + 1] = (jointOrigins[O_Htm_tmp] * m + tmp2[O_Htm_tmp] *
          ROBOT_csi_1) + E_tmp[O_Htm_tmp];
        O_Htm_tmp = 3 * i + 2;
        O_Htm[O_Htm_tmp_0 + 2] = (jointOrigins[O_Htm_tmp] * m + tmp2[O_Htm_tmp] *
          ROBOT_csi_1) + E_tmp[O_Htm_tmp];
        O_Htm[i + 12] = 0.0;
      }

      O_Htm[3] = 0.0;
      O_Htm[7] = 0.0;
      O_Htm[11] = 0.0;
      O_Htm[15] = 1.0;
    }

    ROBOT_g0_size[0] = 4;
    ROBOT_g0_size[1] = 4;
    for (i = 0; i < 4; i++) {
      for (O_Htm_tmp = 0; O_Htm_tmp < 4; O_Htm_tmp++) {
        ROBOT_g0_data[O_Htm_tmp + ROBOT_g0_size[0] * i] = ROBOT_g0
          [(idxStart_4Row + O_Htm_tmp) + 12 * i];
      }
    }

    Planar_robot_3DoF_sim_mtimes(ROBOT_g0_data, ROBOT_g0_size, O_Htm, Htm_data,
      Htm_size);

    /* :  O_Htm = O_Htm_pre * Htm; */
    for (i = 0; i < 4; i++) {
      Iz = O_Htm_pre[i];
      ROBOT_csi_1 = O_Htm_pre[i + 4];
      m = O_Htm_pre[i + 8];
      O_Htm_pre_0 = O_Htm_pre[i + 12];
      for (O_Htm_tmp = 0; O_Htm_tmp < 4; O_Htm_tmp++) {
        O_Htm_tmp_0 = O_Htm_tmp << 2;
        O_Htm_1 = Htm_data[O_Htm_tmp_0] * Iz;
        O_Htm_1 += Htm_data[O_Htm_tmp_0 + 1] * ROBOT_csi_1;
        O_Htm_1 += Htm_data[O_Htm_tmp_0 + 2] * m;
        O_Htm_1 += Htm_data[O_Htm_tmp_0 + 3] * O_Htm_pre_0;
        O_Htm[i + O_Htm_tmp_0] = O_Htm_1;
      }
    }

    /* :  inertialTwist = relTwist; */
    for (i = 0; i < 6; i++) {
      inertialTwist[i] = relTwist[i];
    }

    /* :  invAd = InvAdjoint(Htm); */
    /* :  R_tra = Htm(1:3,1:3).'; */
    for (i = 0; i < 3; i++) {
      jointOrigins[3 * i] = Htm_data[i];
      jointOrigins[3 * i + 1] = Htm_data[i + Htm_size[0]];
      jointOrigins[3 * i + 2] = Htm_data[(Htm_size[0] << 1) + i];
    }

    /* :  p_hat = Skew( Htm(1:3,4) ); */
    /* :  out = [ */
    /* :      0         -vec(3)   vec(2) */
    /* :      vec(3)    0         -vec(1) */
    /* :      -vec(2)   vec(1)    0 */
    /* :      ]; */
    /* :  out = [R_tra -R_tra*p_hat; zeros(3) R_tra]; */
    for (i = 0; i <= 6; i += 2) {
      tmp_2 = _mm_loadu_pd(&jointOrigins[i]);
      tmp_2 = _mm_mul_pd(tmp_2, _mm_set1_pd(-1.0));
      _mm_storeu_pd(&tmp2[i], tmp_2);
    }

    for (i = 8; i < 9; i++) {
      tmp2[i] = -jointOrigins[i];
    }

    tmp[3] = -Htm_data[Htm_size[0] * 3 + 2];
    tmp[6] = Htm_data[Htm_size[0] * 3 + 1];
    tmp[1] = Htm_data[Htm_size[0] * 3 + 2];
    tmp[7] = -Htm_data[Htm_size[0] * 3];
    tmp[2] = -Htm_data[Htm_size[0] * 3 + 1];
    tmp[5] = Htm_data[Htm_size[0] * 3];
    for (i = 0; i < 3; i++) {
      Iz = tmp2[i];
      ROBOT_csi_1 = tmp2[i + 3];
      m = tmp2[i + 6];
      for (O_Htm_tmp = 0; O_Htm_tmp < 3; O_Htm_tmp++) {
        O_Htm_pre_0 = tmp[3 * O_Htm_tmp] * Iz;
        O_Htm_pre_0 += tmp[3 * O_Htm_tmp + 1] * ROBOT_csi_1;
        O_Htm_pre_0 += tmp[3 * O_Htm_tmp + 2] * m;
        S[i + 3 * O_Htm_tmp] = O_Htm_pre_0;
        invAd[O_Htm_tmp + 6 * i] = jointOrigins[3 * i + O_Htm_tmp];
      }
    }

    for (i = 0; i < 3; i++) {
      O_Htm_tmp = (i + 3) * 6;
      invAd[O_Htm_tmp] = S[3 * i];
      invAd[6 * i + 3] = 0.0;
      invAd[O_Htm_tmp + 3] = jointOrigins[3 * i];
      idxStart_4Row = 3 * i + 1;
      invAd[O_Htm_tmp + 1] = S[idxStart_4Row];
      invAd[6 * i + 4] = 0.0;
      invAd[O_Htm_tmp + 4] = jointOrigins[idxStart_4Row];
      idxStart_4Row = 3 * i + 2;
      invAd[O_Htm_tmp + 2] = S[idxStart_4Row];
      invAd[6 * i + 5] = 0.0;
      invAd[O_Htm_tmp + 5] = jointOrigins[idxStart_4Row];
    }

    /* :  J = zeros(6, ndof); */
    /* :  dJ = zeros(6, ndof); */
    std::memset(&J[0], 0, 18U * sizeof(real_T));
    std::memset(&dJ[0], 0, 18U * sizeof(real_T));

    /* :  if iLink > 1 */
    if (ROBOT_Mass_tmp + 1 > 1) {
      /* :  inertialTwist = inertialTwist + invAd * inertialTwist_pre; */
      for (i = 0; i < 6; i++) {
        ROBOT_csi_1 = 0.0;
        for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
          ROBOT_csi_1 += invAd[6 * O_Htm_tmp + i] * inertialTwist_pre[O_Htm_tmp];
        }

        inertialTwist[i] = relTwist[i] + ROBOT_csi_1;
      }

      /* :  J(1:6, 1:iLink-1) = invAd * J_pre(1:6, 1:iLink-1); */
      J_pre_size[0] = 6;
      J_pre_size[1] = ROBOT_Mass_tmp;
      for (i = 0; i < ROBOT_Mass_tmp; i++) {
        for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
          J_pre_data[O_Htm_tmp + 6 * i] = J_pre[6 * i + O_Htm_tmp];
        }
      }

      Planar_robot_3DoF_sim_mtimes_p(invAd, J_pre_data, J_pre_size, tmp_data_1,
        Htm_size);
      O_Htm_tmp_0 = Htm_size[1];
      for (i = 0; i < O_Htm_tmp_0; i++) {
        for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
          J[O_Htm_tmp + 6 * i] = tmp_data_1[6 * i + O_Htm_tmp];
        }
      }

      /* :  dJ(1:6, 1:iLink-1) = invAd * dJ_pre(1:6, 1:iLink-1)- LieBracket(relTwist) * J(1:6, 1:iLink-1); */
      /* :  Sv = Skew( twist(1:3) ); */
      /* :  out = [ */
      /* :      0         -vec(3)   vec(2) */
      /* :      vec(3)    0         -vec(1) */
      /* :      -vec(2)   vec(1)    0 */
      /* :      ]; */
      /* :  Sw = Skew( twist(4:6) ); */
      /* :  out = [ */
      /* :      0         -vec(3)   vec(2) */
      /* :      vec(3)    0         -vec(1) */
      /* :      -vec(2)   vec(1)    0 */
      /* :      ]; */
      jointOrigins[0] = 0.0;
      jointOrigins[3] = -relTwist[5];
      jointOrigins[6] = relTwist[4];
      jointOrigins[1] = relTwist[5];
      jointOrigins[4] = 0.0;
      jointOrigins[7] = -relTwist[3];
      jointOrigins[2] = -relTwist[4];
      jointOrigins[5] = relTwist[3];
      jointOrigins[8] = 0.0;

      /* :  ad = [Sw Sv; zeros(3) Sw]; */
      J_pre_size[0] = 6;
      J_pre_size[1] = ROBOT_Mass_tmp;
      for (i = 0; i < ROBOT_Mass_tmp; i++) {
        for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
          J_pre_data[O_Htm_tmp + 6 * i] = dJ_pre[6 * i + O_Htm_tmp];
        }
      }

      Planar_robot_3DoF_sim_mtimes_p(invAd, J_pre_data, J_pre_size, tmp_data_1,
        Htm_size);
      jointOrigins_0[18] = 0.0;
      jointOrigins_0[24] = -relTwist[2];
      jointOrigins_0[30] = relTwist[1];
      jointOrigins_0[19] = relTwist[2];
      jointOrigins_0[25] = 0.0;
      jointOrigins_0[31] = -relTwist[0];
      jointOrigins_0[20] = -relTwist[1];
      jointOrigins_0[26] = relTwist[0];
      jointOrigins_0[32] = 0.0;
      for (i = 0; i < 3; i++) {
        Iz = jointOrigins[3 * i];
        jointOrigins_0[6 * i] = Iz;
        jointOrigins_0[6 * i + 3] = 0.0;
        O_Htm_tmp = (i + 3) * 6;
        jointOrigins_0[O_Htm_tmp + 3] = Iz;
        Iz = jointOrigins[3 * i + 1];
        jointOrigins_0[6 * i + 1] = Iz;
        jointOrigins_0[6 * i + 4] = 0.0;
        jointOrigins_0[O_Htm_tmp + 4] = Iz;
        Iz = jointOrigins[3 * i + 2];
        jointOrigins_0[6 * i + 2] = Iz;
        jointOrigins_0[6 * i + 5] = 0.0;
        jointOrigins_0[O_Htm_tmp + 5] = Iz;
      }

      J_pre_size[0] = 6;
      J_pre_size[1] = ROBOT_Mass_tmp;
      for (i = 0; i < ROBOT_Mass_tmp; i++) {
        for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
          J_pre_data[O_Htm_tmp + 6 * i] = J[6 * i + O_Htm_tmp];
        }
      }

      Planar_robot_3DoF_sim_mtimes_p(jointOrigins_0, J_pre_data, J_pre_size,
        tmp_data, tmp_size);
      if (Htm_size[1] == tmp_size[1]) {
        J_pre_size[0] = 6;
        J_pre_size[1] = ROBOT_Mass_tmp;
        for (i = 0; i < ROBOT_Mass_tmp; i++) {
          for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
            J_pre_data[O_Htm_tmp + 6 * i] = dJ_pre[6 * i + O_Htm_tmp];
          }
        }

        Planar_robot_3DoF_sim_mtimes_p(invAd, J_pre_data, J_pre_size, tmp_data_1,
          Htm_size);
        jointOrigins_0[18] = 0.0;
        jointOrigins_0[24] = -relTwist[2];
        jointOrigins_0[30] = relTwist[1];
        jointOrigins_0[19] = relTwist[2];
        jointOrigins_0[25] = 0.0;
        jointOrigins_0[31] = -relTwist[0];
        jointOrigins_0[20] = -relTwist[1];
        jointOrigins_0[26] = relTwist[0];
        jointOrigins_0[32] = 0.0;
        for (i = 0; i < 3; i++) {
          Iz = jointOrigins[3 * i];
          jointOrigins_0[6 * i] = Iz;
          jointOrigins_0[6 * i + 3] = 0.0;
          O_Htm_tmp = (i + 3) * 6;
          jointOrigins_0[O_Htm_tmp + 3] = Iz;
          Iz = jointOrigins[3 * i + 1];
          jointOrigins_0[6 * i + 1] = Iz;
          jointOrigins_0[6 * i + 4] = 0.0;
          jointOrigins_0[O_Htm_tmp + 4] = Iz;
          Iz = jointOrigins[3 * i + 2];
          jointOrigins_0[6 * i + 2] = Iz;
          jointOrigins_0[6 * i + 5] = 0.0;
          jointOrigins_0[O_Htm_tmp + 5] = Iz;
        }

        J_pre_size[0] = 6;
        J_pre_size[1] = ROBOT_Mass_tmp;
        for (i = 0; i < ROBOT_Mass_tmp; i++) {
          for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
            J_pre_data[O_Htm_tmp + 6 * i] = J[6 * i + O_Htm_tmp];
          }
        }

        Planar_robot_3DoF_sim_mtimes_p(jointOrigins_0, J_pre_data, J_pre_size,
          tmp_data, tmp_size);
        O_Htm_tmp_0 = Htm_size[1];
        for (i = 0; i < O_Htm_tmp_0; i++) {
          for (O_Htm_tmp = 0; O_Htm_tmp <= 4; O_Htm_tmp += 2) {
            tmp_2 = _mm_loadu_pd(&tmp_data_1[6 * i + O_Htm_tmp]);
            tmp_1 = _mm_loadu_pd(&tmp_data[6 * i + O_Htm_tmp]);
            tmp_2 = _mm_sub_pd(tmp_2, tmp_1);
            _mm_storeu_pd(&dJ[O_Htm_tmp + 6 * i], tmp_2);
          }
        }
      } else {
        Planar_rob_binary_expand_op_ccy(dJ, ROBOT_Mass_tmp, invAd, dJ_pre,
          jointOrigins, relTwist, J);
      }
    }

    /* :  J(1:6, iLink) = relBodyJac; */
    for (i = 0; i < 6; i++) {
      J[i + 6 * ROBOT_Mass_tmp] = ROBOT_csi[idxStart_6Row + i];
    }

    /* :  tmp = J(1:6, 1:iLink)' * ROBOT.Mass(idxStart_6Row:idxEnd_6Row, 1:6); */
    J_size[0] = 6;
    J_size[1] = ROBOT_Mass_tmp + 1;
    for (i = 0; i <= ROBOT_Mass_tmp; i++) {
      for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
        J_pre[O_Htm_tmp + 6 * i] = J[6 * i + O_Htm_tmp];
      }
    }

    ROBOT_Mass_size[0] = 6;
    ROBOT_Mass_size[1] = 6;
    for (i = 0; i < 6; i++) {
      for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
        ROBOT_Mass_data[O_Htm_tmp + ROBOT_Mass_size[0] * i] = ROBOT_Mass
          [(idxStart_6Row + O_Htm_tmp) + 18 * i];
      }
    }

    Planar_robot_3DoF_sim_mtimes_pn(J_pre, J_size, ROBOT_Mass_data,
      ROBOT_Mass_size, dJ_pre, Htm_size);

    /* :  M(1:iLink, 1:iLink) = M(1:iLink, 1:iLink) + tmp * J(1:6, 1:iLink); */
    J_size[0] = 6;
    J_size[1] = ROBOT_Mass_tmp + 1;
    J_size_0[0] = 6;
    J_size_0[1] = ROBOT_Mass_tmp + 1;
    for (i = 0; i <= ROBOT_Mass_tmp; i++) {
      for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
        Iz = J[6 * i + O_Htm_tmp];
        J_pre[O_Htm_tmp + 6 * i] = Iz;
        J_data[O_Htm_tmp + 6 * i] = Iz;
      }
    }

    Planar_robot_3DoF_si_mtimes_pnc(dJ_pre, Htm_size, J_pre, J_size, y, tmp_size);
    Planar_robot_3DoF_si_mtimes_pnc(dJ_pre, Htm_size, J_data, J_size_0, b_y,
      tmp_size_0);
    if ((ROBOT_Mass_tmp + 1 == tmp_size[0]) && (ROBOT_Mass_tmp + 1 ==
         tmp_size_0[1])) {
      J_size[0] = 6;
      J_size[1] = ROBOT_Mass_tmp + 1;
      for (i = 0; i <= ROBOT_Mass_tmp; i++) {
        for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
          J_pre[O_Htm_tmp + 6 * i] = J[6 * i + O_Htm_tmp];
        }
      }

      Planar_robot_3DoF_si_mtimes_pnc(dJ_pre, Htm_size, J_pre, J_size, y,
        tmp_size);
      tmp_size_0[0] = ROBOT_Mass_tmp + 1;
      tmp_size_0[1] = ROBOT_Mass_tmp + 1;
      for (i = 0; i <= ROBOT_Mass_tmp; i++) {
        idxStart_4Row = ((ROBOT_Mass_tmp + 1) / 2) << 1;
        O_Htm_tmp_0 = idxStart_4Row - 2;
        for (O_Htm_tmp = 0; O_Htm_tmp <= O_Htm_tmp_0; O_Htm_tmp += 2) {
          tmp_2 = _mm_loadu_pd(&Planar_robot_3DoF_sim_B.M[3 * i + O_Htm_tmp]);
          tmp_1 = _mm_loadu_pd(&y[tmp_size[0] * i + O_Htm_tmp]);
          tmp_2 = _mm_add_pd(tmp_2, tmp_1);
          _mm_storeu_pd(&b_y[O_Htm_tmp + tmp_size_0[0] * i], tmp_2);
        }

        for (O_Htm_tmp = idxStart_4Row; O_Htm_tmp <= ROBOT_Mass_tmp; O_Htm_tmp++)
        {
          b_y[O_Htm_tmp + tmp_size_0[0] * i] = Planar_robot_3DoF_sim_B.M[3 * i +
            O_Htm_tmp] + y[tmp_size[0] * i + O_Htm_tmp];
        }
      }

      O_Htm_tmp_0 = tmp_size_0[1];
      idxStart_4Row = tmp_size_0[0];
      for (i = 0; i < O_Htm_tmp_0; i++) {
        for (O_Htm_tmp = 0; O_Htm_tmp < idxStart_4Row; O_Htm_tmp++) {
          Planar_robot_3DoF_sim_B.M[O_Htm_tmp + 3 * i] = b_y[tmp_size_0[0] * i +
            O_Htm_tmp];
        }
      }
    } else {
      Planar_robo_binary_expand_op_cc(Planar_robot_3DoF_sim_B.M, ROBOT_Mass_tmp,
        dJ_pre, Htm_size, J);
    }

    /* :  tmptmp = tmp * dJ(1:6, 1:iLink); */
    /* :  dM(1:iLink, 1:iLink) = dM(1:iLink, 1:iLink) + tmptmp + tmptmp'; */
    /* :  Mass_x_ad_tmp = SkewCoriolis(ROBOT.Mass(idxStart_6Row:idxEnd_6Row, 1:6), inertialTwist(4:6)); */
    idxStart_6Row += 3;

    /* :  Sw = Skew(w); */
    /* :  out = [ */
    /* :      0         -vec(3)   vec(2) */
    /* :      vec(3)    0         -vec(1) */
    /* :      -vec(2)   vec(1)    0 */
    /* :      ]; */
    jointOrigins[0] = 0.0;
    jointOrigins[3] = -inertialTwist[5];
    jointOrigins[6] = inertialTwist[4];
    jointOrigins[1] = inertialTwist[5];
    jointOrigins[4] = 0.0;
    jointOrigins[7] = -inertialTwist[3];
    jointOrigins[2] = -inertialTwist[4];
    jointOrigins[5] = inertialTwist[3];
    jointOrigins[8] = 0.0;

    /* :  tmp = Mass(4:6, 1:3)*Sw; */
    /* :  tmp2 = Mass(4:6, 4:6)*Sw; */
    for (i = 0; i < 3; i++) {
      idxStart_4Row = idxStart_6Row + i;
      ROBOT_csi_1 = ROBOT_Mass[idxStart_4Row];
      m = ROBOT_Mass[idxStart_4Row + 54];
      O_Htm_pre_0 = ROBOT_Mass[idxStart_4Row + 18];
      O_Htm_1 = ROBOT_Mass[idxStart_4Row + 72];
      ROBOT_Mass_0 = ROBOT_Mass[idxStart_4Row + 36];
      ROBOT_Mass_1 = ROBOT_Mass[idxStart_4Row + 90];
      for (O_Htm_tmp = 0; O_Htm_tmp < 3; O_Htm_tmp++) {
        Iz = jointOrigins[3 * O_Htm_tmp];
        S_0 = ROBOT_csi_1 * Iz;
        tmp2_0 = m * Iz;
        Iz = jointOrigins[3 * O_Htm_tmp + 1];
        S_0 += O_Htm_pre_0 * Iz;
        tmp2_0 += O_Htm_1 * Iz;
        Iz = jointOrigins[3 * O_Htm_tmp + 2];
        S_0 += ROBOT_Mass_0 * Iz;
        tmp2_0 += ROBOT_Mass_1 * Iz;
        idxStart_4Row = 3 * O_Htm_tmp + i;
        tmp2[idxStart_4Row] = tmp2_0;
        S[idxStart_4Row] = S_0;
      }
    }

    /* :  out = [Mass(1,1)*Sw -tmp'; tmp tmp2 - tmp2']; */
    m = ROBOT_Mass[idxStart_6Row - 3];

    /* :  CC(1:iLink, 1:iLink) = CC(1:iLink, 1:iLink) + J(1:6, 1:iLink)' * Mass_x_ad_tmp * J(1:6, 1:iLink) + tmp * dJ(1:6, 1:iLink); */
    J_size[0] = 6;
    J_size[1] = ROBOT_Mass_tmp + 1;
    for (i = 0; i <= ROBOT_Mass_tmp; i++) {
      for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
        J_pre[O_Htm_tmp + 6 * i] = J[6 * i + O_Htm_tmp];
      }
    }

    for (i = 0; i < 3; i++) {
      invAd[6 * i] = jointOrigins[3 * i] * m;
      idxStart_6Row = (i + 3) * 6;
      invAd[idxStart_6Row] = -S[i];
      invAd[6 * i + 3] = S[3 * i];
      invAd[idxStart_6Row + 3] = tmp2[3 * i] - tmp2[i];
      O_Htm_tmp = 3 * i + 1;
      invAd[6 * i + 1] = jointOrigins[O_Htm_tmp] * m;
      invAd[idxStart_6Row + 1] = -S[i + 3];
      invAd[6 * i + 4] = S[O_Htm_tmp];
      invAd[idxStart_6Row + 4] = tmp2[O_Htm_tmp] - tmp2[i + 3];
      O_Htm_tmp = 3 * i + 2;
      invAd[6 * i + 2] = jointOrigins[O_Htm_tmp] * m;
      invAd[idxStart_6Row + 2] = -S[i + 6];
      invAd[6 * i + 5] = S[O_Htm_tmp];
      invAd[idxStart_6Row + 5] = tmp2[O_Htm_tmp] - tmp2[i + 6];
    }

    Planar_robot_3DoF_s_mtimes_pnc5(J_pre, J_size, invAd, tmp_data_2, tmp_size_4);
    J_size[0] = 6;
    J_size[1] = ROBOT_Mass_tmp + 1;
    J_size_0[0] = 6;
    J_size_0[1] = ROBOT_Mass_tmp + 1;
    for (i = 0; i <= ROBOT_Mass_tmp; i++) {
      for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
        Iz = J[6 * i + O_Htm_tmp];
        J_pre[O_Htm_tmp + 6 * i] = Iz;
        J_data[O_Htm_tmp + 6 * i] = Iz;
      }
    }

    Planar_robot_3DoF_si_mtimes_pnc(tmp_data_2, tmp_size_4, J_pre, J_size, y,
      tmp_size);
    for (i = 0; i < 3; i++) {
      invAd[6 * i] = jointOrigins[3 * i] * m;
      idxStart_6Row = (i + 3) * 6;
      invAd[idxStart_6Row] = -S[i];
      invAd[6 * i + 3] = S[3 * i];
      invAd[idxStart_6Row + 3] = tmp2[3 * i] - tmp2[i];
      O_Htm_tmp = 3 * i + 1;
      invAd[6 * i + 1] = jointOrigins[O_Htm_tmp] * m;
      invAd[idxStart_6Row + 1] = -S[i + 3];
      invAd[6 * i + 4] = S[O_Htm_tmp];
      invAd[idxStart_6Row + 4] = tmp2[O_Htm_tmp] - tmp2[i + 3];
      O_Htm_tmp = 3 * i + 2;
      invAd[6 * i + 2] = jointOrigins[O_Htm_tmp] * m;
      invAd[idxStart_6Row + 2] = -S[i + 6];
      invAd[6 * i + 5] = S[O_Htm_tmp];
      invAd[idxStart_6Row + 5] = tmp2[O_Htm_tmp] - tmp2[i + 6];
    }

    Planar_robot_3DoF_s_mtimes_pnc5(J_data, J_size_0, invAd, tmp_data_2,
      tmp_size_4);
    J_size[0] = 6;
    J_size[1] = ROBOT_Mass_tmp + 1;
    J_size_0[0] = 6;
    J_size_0[1] = ROBOT_Mass_tmp + 1;
    for (i = 0; i <= ROBOT_Mass_tmp; i++) {
      for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
        Iz = J[6 * i + O_Htm_tmp];
        J_pre[O_Htm_tmp + 6 * i] = Iz;
        J_data[O_Htm_tmp + 6 * i] = Iz;
      }
    }

    Planar_robot_3DoF_si_mtimes_pnc(tmp_data_2, tmp_size_4, J_pre, J_size, b_y,
      tmp_size_0);
    for (i = 0; i < 3; i++) {
      invAd[6 * i] = jointOrigins[3 * i] * m;
      idxStart_6Row = (i + 3) * 6;
      invAd[idxStart_6Row] = -S[i];
      invAd[6 * i + 3] = S[3 * i];
      invAd[idxStart_6Row + 3] = tmp2[3 * i] - tmp2[i];
      O_Htm_tmp = 3 * i + 1;
      invAd[6 * i + 1] = jointOrigins[O_Htm_tmp] * m;
      invAd[idxStart_6Row + 1] = -S[i + 3];
      invAd[6 * i + 4] = S[O_Htm_tmp];
      invAd[idxStart_6Row + 4] = tmp2[O_Htm_tmp] - tmp2[i + 3];
      O_Htm_tmp = 3 * i + 2;
      invAd[6 * i + 2] = jointOrigins[O_Htm_tmp] * m;
      invAd[idxStart_6Row + 2] = -S[i + 6];
      invAd[6 * i + 5] = S[O_Htm_tmp];
      invAd[idxStart_6Row + 5] = tmp2[O_Htm_tmp] - tmp2[i + 6];
    }

    Planar_robot_3DoF_s_mtimes_pnc5(J_data, J_size_0, invAd, tmp_data_2,
      tmp_size_4);
    J_size[0] = 6;
    J_size[1] = ROBOT_Mass_tmp + 1;
    for (i = 0; i <= ROBOT_Mass_tmp; i++) {
      for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
        J_pre[O_Htm_tmp + 6 * i] = J[6 * i + O_Htm_tmp];
      }
    }

    Planar_robot_3DoF_si_mtimes_pnc(tmp_data_2, tmp_size_4, J_pre, J_size, b_I,
      tmp_size_1);
    J_size[0] = 6;
    J_size[1] = ROBOT_Mass_tmp + 1;
    for (i = 0; i <= ROBOT_Mass_tmp; i++) {
      for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
        J_pre[O_Htm_tmp + 6 * i] = dJ[6 * i + O_Htm_tmp];
      }
    }

    Planar_robot_3DoF_si_mtimes_pnc(dJ_pre, Htm_size, J_pre, J_size, y,
      tmp_size_2);
    J_size[0] = 6;
    J_size[1] = ROBOT_Mass_tmp + 1;
    for (i = 0; i <= ROBOT_Mass_tmp; i++) {
      for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
        J_pre[O_Htm_tmp + 6 * i] = J[6 * i + O_Htm_tmp];
      }
    }

    for (i = 0; i < 3; i++) {
      invAd[6 * i] = jointOrigins[3 * i] * m;
      idxStart_6Row = (i + 3) * 6;
      invAd[idxStart_6Row] = -S[i];
      invAd[6 * i + 3] = S[3 * i];
      invAd[idxStart_6Row + 3] = tmp2[3 * i] - tmp2[i];
      O_Htm_tmp = 3 * i + 1;
      invAd[6 * i + 1] = jointOrigins[O_Htm_tmp] * m;
      invAd[idxStart_6Row + 1] = -S[i + 3];
      invAd[6 * i + 4] = S[O_Htm_tmp];
      invAd[idxStart_6Row + 4] = tmp2[O_Htm_tmp] - tmp2[i + 3];
      O_Htm_tmp = 3 * i + 2;
      invAd[6 * i + 2] = jointOrigins[O_Htm_tmp] * m;
      invAd[idxStart_6Row + 2] = -S[i + 6];
      invAd[6 * i + 5] = S[O_Htm_tmp];
      invAd[idxStart_6Row + 5] = tmp2[O_Htm_tmp] - tmp2[i + 6];
    }

    Planar_robot_3DoF_s_mtimes_pnc5(J_pre, J_size, invAd, tmp_data_2, tmp_size_4);
    J_size[0] = 6;
    J_size[1] = ROBOT_Mass_tmp + 1;
    for (i = 0; i <= ROBOT_Mass_tmp; i++) {
      for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
        J_pre[O_Htm_tmp + 6 * i] = J[6 * i + O_Htm_tmp];
      }
    }

    Planar_robot_3DoF_si_mtimes_pnc(tmp_data_2, tmp_size_4, J_pre, J_size, y,
      tmp_size_3);
    J_size[0] = 6;
    J_size[1] = ROBOT_Mass_tmp + 1;
    for (i = 0; i <= ROBOT_Mass_tmp; i++) {
      for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
        J_pre[O_Htm_tmp + 6 * i] = dJ[6 * i + O_Htm_tmp];
      }
    }

    Planar_robot_3DoF_si_mtimes_pnc(dJ_pre, Htm_size, J_pre, J_size, y,
      tmp_size_4);
    if ((ROBOT_Mass_tmp + 1 == tmp_size[0]) && (ROBOT_Mass_tmp + 1 ==
         tmp_size_0[1]) && ((ROBOT_Mass_tmp + 1 == 1 ? tmp_size_1[0] :
          ROBOT_Mass_tmp + 1) == tmp_size_2[0]) && ((ROBOT_Mass_tmp + 1 == 1 ?
          tmp_size_3[1] : ROBOT_Mass_tmp + 1) == tmp_size_4[1])) {
      J_size[0] = 6;
      J_size[1] = ROBOT_Mass_tmp + 1;
      for (i = 0; i <= ROBOT_Mass_tmp; i++) {
        for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
          J_pre[O_Htm_tmp + 6 * i] = J[6 * i + O_Htm_tmp];
        }
      }

      for (i = 0; i < 3; i++) {
        invAd[6 * i] = jointOrigins[3 * i] * m;
        idxStart_6Row = (i + 3) * 6;
        invAd[idxStart_6Row] = -S[i];
        invAd[6 * i + 3] = S[3 * i];
        invAd[idxStart_6Row + 3] = tmp2[3 * i] - tmp2[i];
        O_Htm_tmp = 3 * i + 1;
        invAd[6 * i + 1] = jointOrigins[O_Htm_tmp] * m;
        invAd[idxStart_6Row + 1] = -S[i + 3];
        invAd[6 * i + 4] = S[O_Htm_tmp];
        invAd[idxStart_6Row + 4] = tmp2[O_Htm_tmp] - tmp2[i + 3];
        O_Htm_tmp = 3 * i + 2;
        invAd[6 * i + 2] = jointOrigins[O_Htm_tmp] * m;
        invAd[idxStart_6Row + 2] = -S[i + 6];
        invAd[6 * i + 5] = S[O_Htm_tmp];
        invAd[idxStart_6Row + 5] = tmp2[O_Htm_tmp] - tmp2[i + 6];
      }

      Planar_robot_3DoF_s_mtimes_pnc5(J_pre, J_size, invAd, tmp_data_2,
        tmp_size_4);
      J_size[0] = 6;
      J_size[1] = ROBOT_Mass_tmp + 1;
      for (i = 0; i <= ROBOT_Mass_tmp; i++) {
        for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
          J_pre[O_Htm_tmp + 6 * i] = J[6 * i + O_Htm_tmp];
        }
      }

      Planar_robot_3DoF_si_mtimes_pnc(tmp_data_2, tmp_size_4, J_pre, J_size, y,
        tmp_size);
      J_size[0] = 6;
      J_size[1] = ROBOT_Mass_tmp + 1;
      for (i = 0; i <= ROBOT_Mass_tmp; i++) {
        for (O_Htm_tmp = 0; O_Htm_tmp < 6; O_Htm_tmp++) {
          J_pre[O_Htm_tmp + 6 * i] = dJ[6 * i + O_Htm_tmp];
        }
      }

      Planar_robot_3DoF_si_mtimes_pnc(dJ_pre, Htm_size, J_pre, J_size, b_y,
        tmp_size_0);
      tmp_size_1[0] = ROBOT_Mass_tmp + 1;
      tmp_size_1[1] = ROBOT_Mass_tmp + 1;
      for (i = 0; i <= ROBOT_Mass_tmp; i++) {
        idxStart_4Row = ((ROBOT_Mass_tmp + 1) / 2) << 1;
        O_Htm_tmp_0 = idxStart_4Row - 2;
        for (O_Htm_tmp = 0; O_Htm_tmp <= O_Htm_tmp_0; O_Htm_tmp += 2) {
          tmp_2 = _mm_loadu_pd(&Planar_robot_3DoF_sim_B.CC[3 * i + O_Htm_tmp]);
          tmp_1 = _mm_loadu_pd(&y[tmp_size[0] * i + O_Htm_tmp]);
          tmp_2 = _mm_add_pd(tmp_2, tmp_1);
          tmp_1 = _mm_loadu_pd(&b_y[tmp_size_0[0] * i + O_Htm_tmp]);
          tmp_2 = _mm_add_pd(tmp_2, tmp_1);
          _mm_storeu_pd(&b_I[O_Htm_tmp + tmp_size_1[0] * i], tmp_2);
        }

        for (O_Htm_tmp = idxStart_4Row; O_Htm_tmp <= ROBOT_Mass_tmp; O_Htm_tmp++)
        {
          b_I[O_Htm_tmp + tmp_size_1[0] * i] = (Planar_robot_3DoF_sim_B.CC[3 * i
            + O_Htm_tmp] + y[tmp_size[0] * i + O_Htm_tmp]) + b_y[tmp_size_0[0] *
            i + O_Htm_tmp];
        }
      }

      O_Htm_tmp_0 = tmp_size_1[1];
      idxStart_4Row = tmp_size_1[0];
      for (i = 0; i < O_Htm_tmp_0; i++) {
        for (O_Htm_tmp = 0; O_Htm_tmp < idxStart_4Row; O_Htm_tmp++) {
          Planar_robot_3DoF_sim_B.CC[O_Htm_tmp + 3 * i] = b_I[tmp_size_1[0] * i
            + O_Htm_tmp];
        }
      }
    } else {
      Planar_robot_binary_expand_op_c(Planar_robot_3DoF_sim_B.CC, ROBOT_Mass_tmp,
        J, m, jointOrigins, S, tmp2, dJ_pre, Htm_size, dJ);
    }

    /* :  g(1:iLink) = g(1:iLink) + tmp(1:iLink, 1:3) * O_Htm(1:3, 1:3)' * ROBOT.g_vec; */
    tmp_size_5[0] = ROBOT_Mass_tmp + 1;
    tmp_size_5[1] = 3;
    for (i = 0; i < 3; i++) {
      for (O_Htm_tmp = 0; O_Htm_tmp <= ROBOT_Mass_tmp; O_Htm_tmp++) {
        y[O_Htm_tmp + tmp_size_5[0] * i] = dJ_pre[Htm_size[0] * i + O_Htm_tmp];
      }

      O_Htm_tmp_0 = i << 2;
      tmp2[3 * i] = O_Htm[O_Htm_tmp_0];
      tmp2[3 * i + 1] = O_Htm[O_Htm_tmp_0 + 1];
      tmp2[3 * i + 2] = O_Htm[O_Htm_tmp_0 + 2];
    }

    Planar_robot_3DoF__mtimes_pnc5a(y, tmp_size_5, tmp2, b_y, tmp_size);
    Planar_robot_3DoF_mtimes_pnc5ag(b_y, tmp_size, tmp_data_0, &i);
    if (ROBOT_Mass_tmp + 1 == i) {
      tmp_size_5[0] = ROBOT_Mass_tmp + 1;
      tmp_size_5[1] = 3;
      for (i = 0; i < 3; i++) {
        for (O_Htm_tmp = 0; O_Htm_tmp <= ROBOT_Mass_tmp; O_Htm_tmp++) {
          y[O_Htm_tmp + tmp_size_5[0] * i] = dJ_pre[Htm_size[0] * i + O_Htm_tmp];
        }

        O_Htm_tmp_0 = i << 2;
        tmp2[3 * i] = O_Htm[O_Htm_tmp_0];
        tmp2[3 * i + 1] = O_Htm[O_Htm_tmp_0 + 1];
        tmp2[3 * i + 2] = O_Htm[O_Htm_tmp_0 + 2];
      }

      Planar_robot_3DoF__mtimes_pnc5a(y, tmp_size_5, tmp2, b_y, tmp_size);
      Planar_robot_3DoF_mtimes_pnc5ag(b_y, tmp_size, tmp_data_0, &i);
      idxStart_6Row = ROBOT_Mass_tmp + 1;
      idxStart_4Row = (idxStart_6Row / 2) << 1;
      O_Htm_tmp_0 = idxStart_4Row - 2;
      for (i = 0; i <= O_Htm_tmp_0; i += 2) {
        tmp_2 = _mm_loadu_pd(&Planar_robot_3DoF_sim_B.g[i]);
        tmp_1 = _mm_loadu_pd(&tmp_data_0[i]);
        tmp_2 = _mm_add_pd(tmp_2, tmp_1);
        _mm_storeu_pd(&tmp_data_3[i], tmp_2);
      }

      for (i = idxStart_4Row; i < idxStart_6Row; i++) {
        tmp_data_3[i] = Planar_robot_3DoF_sim_B.g[i] + tmp_data_0[i];
      }

      O_Htm_tmp_0 = idxStart_6Row;
      std::memcpy(&Planar_robot_3DoF_sim_B.g[0], &tmp_data_3[0],
                  static_cast<uint32_T>(O_Htm_tmp_0) * sizeof(real_T));
    } else {
      Planar_robot_3_binary_expand_op(Planar_robot_3DoF_sim_B.g, ROBOT_Mass_tmp,
        dJ_pre, Htm_size, O_Htm);
    }

    /* :  J_pre = J; */
    /* :  dJ_pre = dJ; */
    std::memcpy(&J_pre[0], &J[0], 18U * sizeof(real_T));
    std::memcpy(&dJ_pre[0], &dJ[0], 18U * sizeof(real_T));

    /* :  inertialTwist_pre = inertialTwist; */
    for (i = 0; i < 6; i++) {
      inertialTwist_pre[i] = inertialTwist[i];
    }

    /* :  O_Htm_pre = O_Htm; */
    std::memcpy(&O_Htm_pre[0], &O_Htm[0], sizeof(real_T) << 4U);
  }

  /* :  Htm_TCP = O_Htm(1:4,1:4) * ROBOT.tcp_t_ee; */
  /* :  pos_tcp = Htm_TCP(1:3,4); */
  for (i = 0; i < 3; i++) {
    ROBOT_Mass_tmp = i << 2;
    O_Htm_pre[ROBOT_Mass_tmp] = E_tmp[3 * i];
    O_Htm_pre[ROBOT_Mass_tmp + 1] = E_tmp[3 * i + 1];
    O_Htm_pre[ROBOT_Mass_tmp + 2] = E_tmp[3 * i + 2];
  }

  O_Htm_pre[12] = 0.1;
  O_Htm_pre[13] = 0.0;
  O_Htm_pre[14] = 0.0;
  O_Htm_pre[3] = 0.0;
  O_Htm_pre[7] = 0.0;
  O_Htm_pre[11] = 0.0;
  O_Htm_pre[15] = 1.0;
  for (i = 0; i < 4; i++) {
    ROBOT_Mass_tmp = i << 2;
    Iz = O_Htm_pre[ROBOT_Mass_tmp];
    ROBOT_csi_1 = O_Htm_pre[ROBOT_Mass_tmp + 1];
    m = O_Htm_pre[ROBOT_Mass_tmp + 2];
    O_Htm_pre_0 = O_Htm_pre[ROBOT_Mass_tmp + 3];
    for (O_Htm_tmp = 0; O_Htm_tmp <= 2; O_Htm_tmp += 2) {
      tmp_2 = _mm_loadu_pd(&O_Htm[O_Htm_tmp]);
      tmp_2 = _mm_mul_pd(_mm_set1_pd(Iz), tmp_2);
      tmp_1 = _mm_loadu_pd(&O_Htm[O_Htm_tmp + 4]);
      tmp_1 = _mm_mul_pd(_mm_set1_pd(ROBOT_csi_1), tmp_1);
      tmp_2 = _mm_add_pd(tmp_1, tmp_2);
      tmp_1 = _mm_loadu_pd(&O_Htm[O_Htm_tmp + 8]);
      tmp_1 = _mm_mul_pd(_mm_set1_pd(m), tmp_1);
      tmp_2 = _mm_add_pd(tmp_1, tmp_2);
      tmp_1 = _mm_loadu_pd(&O_Htm[O_Htm_tmp + 12]);
      tmp_1 = _mm_mul_pd(_mm_set1_pd(O_Htm_pre_0), tmp_1);
      tmp_2 = _mm_add_pd(tmp_1, tmp_2);
      _mm_storeu_pd(&O_Htm_0[O_Htm_tmp + ROBOT_Mass_tmp], tmp_2);
    }
  }

  /* :  B_J_TCP = InvAdjoint(ROBOT.tcp_t_ee) * J(1:6, 1:ndof); */
  /* :  O_J_TCP = [Htm_TCP(1:3, 1:3)* B_J_TCP(1:3, 1:ndof);Htm_TCP(1:3, 1:3)*B_J_TCP(4:6, 1:ndof)]; */
  Iz = O_Htm_0[12];
  Planar_robot_3DoF_sim_B.pos_tcp[0] = Iz;

  /* SampleTimeMath: '<S2>/TSamp'
   *
   * About '<S2>/TSamp':
   *  y = u * K where K = 1 / ( w * Ts )
   */
  Iz *= 1000.0;

  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
  Planar_robot_3DoF_sim_B.TSamp[0] = Iz;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* UnitDelay: '<S2>/UD' */
  ROBOT_csi_1 = Planar_robot_3DoF_sim_DW.UD_DSTATE[0];

  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
  Planar_robot_3DoF_sim_B.Uk1[0] = ROBOT_csi_1;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* Sum: '<S2>/Diff' */
  Iz -= ROBOT_csi_1;

  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
  Planar_robot_3DoF_sim_B.Diff[0] = Iz;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* SampleTimeMath: '<S3>/TSamp'
   *
   * About '<S3>/TSamp':
   *  y = u * K where K = 1 / ( w * Ts )
   */
  Iz *= 1000.0;

  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
  Planar_robot_3DoF_sim_B.TSamp_g[0] = Iz;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* UnitDelay: '<S3>/UD' */
  ROBOT_csi_1 = Planar_robot_3DoF_sim_DW.UD_DSTATE_o[0];

  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
  Planar_robot_3DoF_sim_B.Uk1_o[0] = ROBOT_csi_1;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* Sum: '<S3>/Diff' */
  Planar_robot_3DoF_sim_B.Diff_d[0] = Iz - ROBOT_csi_1;

  /* MATLAB Function: '<S1>/calcSysMatrices' */
  Iz = O_Htm_0[13];
  Planar_robot_3DoF_sim_B.pos_tcp[1] = Iz;

  /* SampleTimeMath: '<S2>/TSamp'
   *
   * About '<S2>/TSamp':
   *  y = u * K where K = 1 / ( w * Ts )
   */
  Iz *= 1000.0;

  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
  Planar_robot_3DoF_sim_B.TSamp[1] = Iz;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* UnitDelay: '<S2>/UD' */
  ROBOT_csi_1 = Planar_robot_3DoF_sim_DW.UD_DSTATE[1];

  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
  Planar_robot_3DoF_sim_B.Uk1[1] = ROBOT_csi_1;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* Sum: '<S2>/Diff' */
  Iz -= ROBOT_csi_1;

  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
  Planar_robot_3DoF_sim_B.Diff[1] = Iz;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* SampleTimeMath: '<S3>/TSamp'
   *
   * About '<S3>/TSamp':
   *  y = u * K where K = 1 / ( w * Ts )
   */
  Iz *= 1000.0;

  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
  Planar_robot_3DoF_sim_B.TSamp_g[1] = Iz;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* UnitDelay: '<S3>/UD' */
  ROBOT_csi_1 = Planar_robot_3DoF_sim_DW.UD_DSTATE_o[1];

  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
  Planar_robot_3DoF_sim_B.Uk1_o[1] = ROBOT_csi_1;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* Sum: '<S3>/Diff' */
  Planar_robot_3DoF_sim_B.Diff_d[1] = Iz - ROBOT_csi_1;

  /* MATLAB Function: '<S1>/calcSysMatrices' */
  Iz = O_Htm_0[14];
  Planar_robot_3DoF_sim_B.pos_tcp[2] = Iz;

  /* SampleTimeMath: '<S2>/TSamp'
   *
   * About '<S2>/TSamp':
   *  y = u * K where K = 1 / ( w * Ts )
   */
  Iz *= 1000.0;

  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
  Planar_robot_3DoF_sim_B.TSamp[2] = Iz;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* UnitDelay: '<S2>/UD' */
  ROBOT_csi_1 = Planar_robot_3DoF_sim_DW.UD_DSTATE[2];

  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
  Planar_robot_3DoF_sim_B.Uk1[2] = ROBOT_csi_1;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* Sum: '<S2>/Diff' */
  Iz -= ROBOT_csi_1;

  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
  Planar_robot_3DoF_sim_B.Diff[2] = Iz;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* SampleTimeMath: '<S3>/TSamp'
   *
   * About '<S3>/TSamp':
   *  y = u * K where K = 1 / ( w * Ts )
   */
  Iz *= 1000.0;

  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
  Planar_robot_3DoF_sim_B.TSamp_g[2] = Iz;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* UnitDelay: '<S3>/UD' */
  ROBOT_csi_1 = Planar_robot_3DoF_sim_DW.UD_DSTATE_o[2];

  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
  Planar_robot_3DoF_sim_B.Uk1_o[2] = ROBOT_csi_1;

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* Sum: '<S3>/Diff' */
  Planar_robot_3DoF_sim_B.Diff_d[2] = Iz - ROBOT_csi_1;

  /* Product: '<S1>/Product5' */
  rt_invd3x3_snf(Planar_robot_3DoF_sim_B.M, Planar_robot_3DoF_sim_B.Product5);

  /* Product: '<S1>/Product3' */
  std::memcpy(&E_tmp[0], &Planar_robot_3DoF_sim_B.CC[0], 9U * sizeof(real_T));
  Iz = Planar_robot_3DoF_sim_B.dq[0];
  m = Planar_robot_3DoF_sim_B.dq[1];
  O_Htm_pre_0 = Planar_robot_3DoF_sim_B.dq[2];

  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
  for (i = 0; i <= 0; i += 2) {
    /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
    /* Product: '<S1>/Product3' */
    tmp_2 = _mm_loadu_pd(&E_tmp[i]);
    tmp_2 = _mm_mul_pd(tmp_2, _mm_set1_pd(Iz));
    tmp_1 = _mm_loadu_pd(&E_tmp[i + 3]);
    tmp_1 = _mm_mul_pd(tmp_1, _mm_set1_pd(m));

    /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
    tmp_2 = _mm_add_pd(tmp_1, tmp_2);

    /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
    /* Product: '<S1>/Product3' */
    tmp_1 = _mm_loadu_pd(&E_tmp[i + 6]);
    tmp_1 = _mm_mul_pd(tmp_1, _mm_set1_pd(O_Htm_pre_0));

    /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
    tmp_2 = _mm_add_pd(tmp_1, tmp_2);

    /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
    /* Product: '<S1>/Product3' */
    _mm_storeu_pd(&Planar_robot_3DoF_sim_B.Product3[i], tmp_2);

    /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
  }

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  for (i = 2; i < 3; i++) {
    /* Product: '<S1>/Product3' */
    ROBOT_csi_1 = E_tmp[i] * Iz;
    ROBOT_csi_1 += E_tmp[i + 3] * m;
    ROBOT_csi_1 += E_tmp[i + 6] * O_Htm_pre_0;

    /* Product: '<S1>/Product3' */
    Planar_robot_3DoF_sim_B.Product3[i] = ROBOT_csi_1;
  }

  /* Sum: '<S1>/Sum4' */
  Planar_robot_3DoF_sim_B.Sum4[0] = (Planar_robot_3DoF_sim_B.j1_torque -
    Planar_robot_3DoF_sim_B.Product3[0]) - Planar_robot_3DoF_sim_B.g[0];
  Planar_robot_3DoF_sim_B.Sum4[1] = (Planar_robot_3DoF_sim_B.j2_torque -
    Planar_robot_3DoF_sim_B.Product3[1]) - Planar_robot_3DoF_sim_B.g[1];
  Planar_robot_3DoF_sim_B.Sum4[2] = (Planar_robot_3DoF_sim_B.j3_torque -
    Planar_robot_3DoF_sim_B.Product3[2]) - Planar_robot_3DoF_sim_B.g[2];

  /* Product: '<S1>/Product1' incorporates:
   *  Product: '<S1>/Product5'
   */
  std::memcpy(&E_tmp[0], &Planar_robot_3DoF_sim_B.Product5[0], 9U * sizeof
              (real_T));
  Iz = Planar_robot_3DoF_sim_B.Sum4[0];
  m = Planar_robot_3DoF_sim_B.Sum4[1];
  O_Htm_pre_0 = Planar_robot_3DoF_sim_B.Sum4[2];

  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
  for (i = 0; i <= 0; i += 2) {
    /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
    /* Product: '<S1>/Product1' */
    tmp_2 = _mm_loadu_pd(&E_tmp[i]);
    tmp_2 = _mm_mul_pd(tmp_2, _mm_set1_pd(Iz));
    tmp_1 = _mm_loadu_pd(&E_tmp[i + 3]);
    tmp_1 = _mm_mul_pd(tmp_1, _mm_set1_pd(m));

    /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
    tmp_2 = _mm_add_pd(tmp_1, tmp_2);

    /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
    /* Product: '<S1>/Product1' */
    tmp_1 = _mm_loadu_pd(&E_tmp[i + 6]);
    tmp_1 = _mm_mul_pd(tmp_1, _mm_set1_pd(O_Htm_pre_0));

    /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
    tmp_2 = _mm_add_pd(tmp_1, tmp_2);

    /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
    /* Product: '<S1>/Product1' */
    _mm_storeu_pd(&Planar_robot_3DoF_sim_B.Product1[i], tmp_2);

    /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
  }

  /* Outputs for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  for (i = 2; i < 3; i++) {
    /* Product: '<S1>/Product1' */
    ROBOT_csi_1 = E_tmp[i] * Iz;
    ROBOT_csi_1 += E_tmp[i + 3] * m;
    ROBOT_csi_1 += E_tmp[i + 6] * O_Htm_pre_0;

    /* Product: '<S1>/Product1' */
    Planar_robot_3DoF_sim_B.Product1[i] = ROBOT_csi_1;
  }

  /* End of Outputs for SubSystem: '<Root>/planar_robot3dof_FD' */
  /* SignalConversion generated from: '<Root>/To Workspace' */
  Planar_robot_3DoF_sim_B.TmpSignalConversionAtToWorkspac[0] =
    Planar_robot_3DoF_sim_B.q[0];
  Planar_robot_3DoF_sim_B.TmpSignalConversionAtToWorkspac[3] =
    Planar_robot_3DoF_sim_B.dq[0];
  Planar_robot_3DoF_sim_B.TmpSignalConversionAtToWorkspac[6] =
    Planar_robot_3DoF_sim_B.Product1[0];
  Planar_robot_3DoF_sim_B.TmpSignalConversionAtToWorkspac[9] =
    Planar_robot_3DoF_sim_B.pos_tcp[0];
  Planar_robot_3DoF_sim_B.TmpSignalConversionAtToWorkspac[12] =
    Planar_robot_3DoF_sim_B.Diff[0];
  Planar_robot_3DoF_sim_B.TmpSignalConversionAtToWorkspac[15] =
    Planar_robot_3DoF_sim_B.Diff_d[0];
  Planar_robot_3DoF_sim_B.TmpSignalConversionAtToWorkspac[1] =
    Planar_robot_3DoF_sim_B.q[1];
  Planar_robot_3DoF_sim_B.TmpSignalConversionAtToWorkspac[4] =
    Planar_robot_3DoF_sim_B.dq[1];
  Planar_robot_3DoF_sim_B.TmpSignalConversionAtToWorkspac[7] =
    Planar_robot_3DoF_sim_B.Product1[1];
  Planar_robot_3DoF_sim_B.TmpSignalConversionAtToWorkspac[10] =
    Planar_robot_3DoF_sim_B.pos_tcp[1];
  Planar_robot_3DoF_sim_B.TmpSignalConversionAtToWorkspac[13] =
    Planar_robot_3DoF_sim_B.Diff[1];
  Planar_robot_3DoF_sim_B.TmpSignalConversionAtToWorkspac[16] =
    Planar_robot_3DoF_sim_B.Diff_d[1];
  Planar_robot_3DoF_sim_B.TmpSignalConversionAtToWorkspac[2] =
    Planar_robot_3DoF_sim_B.q[2];
  Planar_robot_3DoF_sim_B.TmpSignalConversionAtToWorkspac[5] =
    Planar_robot_3DoF_sim_B.dq[2];
  Planar_robot_3DoF_sim_B.TmpSignalConversionAtToWorkspac[8] =
    Planar_robot_3DoF_sim_B.Product1[2];
  Planar_robot_3DoF_sim_B.TmpSignalConversionAtToWorkspac[11] =
    Planar_robot_3DoF_sim_B.pos_tcp[2];
  Planar_robot_3DoF_sim_B.TmpSignalConversionAtToWorkspac[14] =
    Planar_robot_3DoF_sim_B.Diff[2];
  Planar_robot_3DoF_sim_B.TmpSignalConversionAtToWorkspac[17] =
    Planar_robot_3DoF_sim_B.Diff_d[2];

  /* ToWorkspace: '<Root>/To Workspace' */
  rt_UpdateStructLogVar((StructLogVar *)
                        Planar_robot_3DoF_sim_DW.ToWorkspace_PWORK.LoggedData,
                        (nullptr),
                        &Planar_robot_3DoF_sim_B.TmpSignalConversionAtToWorkspac[
                        0]);

  /* ToWorkspace: '<Root>/To Workspace1' */
  rt_UpdateLogVar((LogVar *)(LogVar*)
                  (Planar_robot_3DoF_sim_DW.ToWorkspace1_PWORK.LoggedData),
                  &Planar_robot_3DoF_sim_B.joint_torque[0], 0);

  /* Matfile logging */
  rt_UpdateTXYLogVars((&Planar_robot_3DoF_sim_M)->rtwLogInfo,
                      ((&Planar_robot_3DoF_sim_M)->Timing.t));

  /* Update for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* Update for DiscreteIntegrator: '<S1>/Discrete-Time Integrator1' */
  Planar_robot_3DoF_sim_DW.DiscreteTimeIntegrator1_DSTATE[0] += 0.001 *
    Planar_robot_3DoF_sim_B.dq[0];

  /* Update for DiscreteIntegrator: '<S1>/Discrete-Time Integrator' */
  Planar_robot_3DoF_sim_DW.DiscreteTimeIntegrator_DSTATE[0] += 0.001 *
    Planar_robot_3DoF_sim_B.Product1[0];

  /* Update for UnitDelay: '<S2>/UD' */
  Planar_robot_3DoF_sim_DW.UD_DSTATE[0] = Planar_robot_3DoF_sim_B.TSamp[0];

  /* Update for UnitDelay: '<S3>/UD' */
  Planar_robot_3DoF_sim_DW.UD_DSTATE_o[0] = Planar_robot_3DoF_sim_B.TSamp_g[0];

  /* Update for DiscreteIntegrator: '<S1>/Discrete-Time Integrator1' */
  Planar_robot_3DoF_sim_DW.DiscreteTimeIntegrator1_DSTATE[1] += 0.001 *
    Planar_robot_3DoF_sim_B.dq[1];

  /* Update for DiscreteIntegrator: '<S1>/Discrete-Time Integrator' */
  Planar_robot_3DoF_sim_DW.DiscreteTimeIntegrator_DSTATE[1] += 0.001 *
    Planar_robot_3DoF_sim_B.Product1[1];

  /* Update for UnitDelay: '<S2>/UD' */
  Planar_robot_3DoF_sim_DW.UD_DSTATE[1] = Planar_robot_3DoF_sim_B.TSamp[1];

  /* Update for UnitDelay: '<S3>/UD' */
  Planar_robot_3DoF_sim_DW.UD_DSTATE_o[1] = Planar_robot_3DoF_sim_B.TSamp_g[1];

  /* Update for DiscreteIntegrator: '<S1>/Discrete-Time Integrator1' */
  Planar_robot_3DoF_sim_DW.DiscreteTimeIntegrator1_DSTATE[2] += 0.001 *
    Planar_robot_3DoF_sim_B.dq[2];

  /* Update for DiscreteIntegrator: '<S1>/Discrete-Time Integrator' */
  Planar_robot_3DoF_sim_DW.DiscreteTimeIntegrator_DSTATE[2] += 0.001 *
    Planar_robot_3DoF_sim_B.Product1[2];

  /* Update for UnitDelay: '<S2>/UD' */
  Planar_robot_3DoF_sim_DW.UD_DSTATE[2] = Planar_robot_3DoF_sim_B.TSamp[2];

  /* Update for UnitDelay: '<S3>/UD' */
  Planar_robot_3DoF_sim_DW.UD_DSTATE_o[2] = Planar_robot_3DoF_sim_B.TSamp_g[2];

  /* End of Update for SubSystem: '<Root>/planar_robot3dof_FD' */

  /* signal main to stop simulation */
  {                                    /* Sample time: [0.0s, 0.0s] */
    if ((rtmGetTFinal((&Planar_robot_3DoF_sim_M))!=-1) &&
        !((rtmGetTFinal((&Planar_robot_3DoF_sim_M))-(&Planar_robot_3DoF_sim_M)
           ->Timing.t[0]) > (&Planar_robot_3DoF_sim_M)->Timing.t[0] *
          (DBL_EPSILON))) {
      rtmSetErrorStatus((&Planar_robot_3DoF_sim_M), "Simulation finished");
    }
  }

  /* Update absolute time for base rate */
  /* The "clockTick0" counts the number of times the code of this task has
   * been executed. The absolute time is the multiplication of "clockTick0"
   * and "Timing.stepSize0". Size of "clockTick0" ensures timer will not
   * overflow during the application lifespan selected.
   * Timer of this task consists of two 32 bit unsigned integers.
   * The two integers represent the low bits Timing.clockTick0 and the high bits
   * Timing.clockTickH0. When the low bit overflows to 0, the high bits increment.
   */
  if (!(++(&Planar_robot_3DoF_sim_M)->Timing.clockTick0)) {
    ++(&Planar_robot_3DoF_sim_M)->Timing.clockTickH0;
  }

  (&Planar_robot_3DoF_sim_M)->Timing.t[0] = (&Planar_robot_3DoF_sim_M)
    ->Timing.clockTick0 * (&Planar_robot_3DoF_sim_M)->Timing.stepSize0 +
    (&Planar_robot_3DoF_sim_M)->Timing.clockTickH0 * (&Planar_robot_3DoF_sim_M
    )->Timing.stepSize0 * 4294967296.0;

  {
    /* Update absolute timer for sample time: [0.001s, 0.0s] */
    /* The "clockTick1" counts the number of times the code of this task has
     * been executed. The resolution of this integer timer is 0.001, which is the step size
     * of the task. Size of "clockTick1" ensures timer will not overflow during the
     * application lifespan selected.
     * Timer of this task consists of two 32 bit unsigned integers.
     * The two integers represent the low bits Timing.clockTick1 and the high bits
     * Timing.clockTickH1. When the low bit overflows to 0, the high bits increment.
     */
    (&Planar_robot_3DoF_sim_M)->Timing.clockTick1++;
    if (!(&Planar_robot_3DoF_sim_M)->Timing.clockTick1) {
      (&Planar_robot_3DoF_sim_M)->Timing.clockTickH1++;
    }
  }
}

/* Model initialize function */
void Planar_robot_3DoF_sim::initialize()
{
  /* Registration code */

  /* initialize non-finites */
  rt_InitInfAndNaN(sizeof(real_T));

  {
    /* Setup solver object */
    rtsiSetSimTimeStepPtr(&(&Planar_robot_3DoF_sim_M)->solverInfo,
                          &(&Planar_robot_3DoF_sim_M)->Timing.simTimeStep);
    rtsiSetTPtr(&(&Planar_robot_3DoF_sim_M)->solverInfo, &rtmGetTPtr
                ((&Planar_robot_3DoF_sim_M)));
    rtsiSetStepSizePtr(&(&Planar_robot_3DoF_sim_M)->solverInfo,
                       &(&Planar_robot_3DoF_sim_M)->Timing.stepSize0);
    rtsiSetErrorStatusPtr(&(&Planar_robot_3DoF_sim_M)->solverInfo,
                          (&rtmGetErrorStatus((&Planar_robot_3DoF_sim_M))));
    rtsiSetRTModelPtr(&(&Planar_robot_3DoF_sim_M)->solverInfo,
                      (&Planar_robot_3DoF_sim_M));
  }

  rtsiSetSimTimeStep(&(&Planar_robot_3DoF_sim_M)->solverInfo, MAJOR_TIME_STEP);
  rtsiSetSolverName(&(&Planar_robot_3DoF_sim_M)->solverInfo,"FixedStepDiscrete");
  rtmSetTPtr((&Planar_robot_3DoF_sim_M), &(&Planar_robot_3DoF_sim_M)
             ->Timing.tArray[0]);
  rtmSetTFinal((&Planar_robot_3DoF_sim_M), 5.0);
  (&Planar_robot_3DoF_sim_M)->Timing.stepSize0 = 0.001;

  /* Setup for data logging */
  {
    static RTWLogInfo rt_DataLoggingInfo;
    rt_DataLoggingInfo.loggingInterval = (nullptr);
    (&Planar_robot_3DoF_sim_M)->rtwLogInfo = &rt_DataLoggingInfo;
  }

  /* Setup for data logging */
  {
    rtliSetLogXSignalInfo((&Planar_robot_3DoF_sim_M)->rtwLogInfo, (nullptr));
    rtliSetLogXSignalPtrs((&Planar_robot_3DoF_sim_M)->rtwLogInfo, (nullptr));
    rtliSetLogT((&Planar_robot_3DoF_sim_M)->rtwLogInfo, "");
    rtliSetLogX((&Planar_robot_3DoF_sim_M)->rtwLogInfo, "");
    rtliSetLogXFinal((&Planar_robot_3DoF_sim_M)->rtwLogInfo, "");
    rtliSetLogVarNameModifier((&Planar_robot_3DoF_sim_M)->rtwLogInfo, "rt_");
    rtliSetLogFormat((&Planar_robot_3DoF_sim_M)->rtwLogInfo, 1);
    rtliSetLogMaxRows((&Planar_robot_3DoF_sim_M)->rtwLogInfo, 0);
    rtliSetLogDecimation((&Planar_robot_3DoF_sim_M)->rtwLogInfo, 1);
    rtliSetLogY((&Planar_robot_3DoF_sim_M)->rtwLogInfo, "");
    rtliSetLogYSignalInfo((&Planar_robot_3DoF_sim_M)->rtwLogInfo, (nullptr));
    rtliSetLogYSignalPtrs((&Planar_robot_3DoF_sim_M)->rtwLogInfo, (nullptr));
  }

  /* Matfile logging */
  rt_StartDataLoggingWithStartTime((&Planar_robot_3DoF_sim_M)->rtwLogInfo, 0.0,
    rtmGetTFinal((&Planar_robot_3DoF_sim_M)), (&Planar_robot_3DoF_sim_M)
    ->Timing.stepSize0, (&rtmGetErrorStatus((&Planar_robot_3DoF_sim_M))));

  /* SetupRuntimeResources for ToWorkspace: '<Root>/To Workspace' */
  {
    static int_T rt_ToWksWidths[] { 18 };

    static int_T rt_ToWksNumDimensions[] { 1 };

    static int_T rt_ToWksDimensions[] { 18 };

    static boolean_T rt_ToWksIsVarDims[] { 0 };

    static void *rt_ToWksCurrSigDims[]{ (nullptr) };

    static int_T rt_ToWksCurrSigDimsSize[]{ 4 };

    static BuiltInDTypeId rt_ToWksDataTypeIds[] { SS_DOUBLE };

    static int_T rt_ToWksComplexSignals[]{ 0 };

    static int_T rt_ToWksFrameData[] { 0 };

    static RTWPreprocessingFcnPtr rt_ToWksLoggingPreprocessingFcnPtrs[] {
      (nullptr)
    };

    static const char_T *rt_ToWksLabels[] = { "" };

    static RTWLogSignalInfo rt_ToWksSignalInfo {
      1,
      rt_ToWksWidths,
      rt_ToWksNumDimensions,
      rt_ToWksDimensions,
      rt_ToWksIsVarDims,
      rt_ToWksCurrSigDims,
      rt_ToWksCurrSigDimsSize,
      rt_ToWksDataTypeIds,
      rt_ToWksComplexSignals,
      rt_ToWksFrameData,
      rt_ToWksLoggingPreprocessingFcnPtrs,

      { rt_ToWksLabels },
      (nullptr),
      (nullptr),
      (nullptr),

      { (nullptr) },

      { (nullptr) },
      (nullptr),
      (nullptr)
    };

    static const char_T rt_ToWksBlockName[] {
      "Planar_robot_3DoF_sim/To Workspace" };

    Planar_robot_3DoF_sim_DW.ToWorkspace_PWORK.LoggedData =
      rt_CreateStructLogVar(
      (&Planar_robot_3DoF_sim_M)->rtwLogInfo,
      0.0,
      rtmGetTFinal((&Planar_robot_3DoF_sim_M)),
      (&Planar_robot_3DoF_sim_M)->Timing.stepSize0,
      (&rtmGetErrorStatus((&Planar_robot_3DoF_sim_M))),
      "simout",
      0,
      0,
      1,
      0.001,
      &rt_ToWksSignalInfo,
      rt_ToWksBlockName);
    if (Planar_robot_3DoF_sim_DW.ToWorkspace_PWORK.LoggedData == (nullptr))
      return;
  }

  /* SetupRuntimeResources for ToWorkspace: '<Root>/To Workspace1' */
  {
    int_T dimensions[1]{ 3 };

    Planar_robot_3DoF_sim_DW.ToWorkspace1_PWORK.LoggedData = rt_CreateLogVar(
      (&Planar_robot_3DoF_sim_M)->rtwLogInfo,
      0.0,
      rtmGetTFinal((&Planar_robot_3DoF_sim_M)),
      (&Planar_robot_3DoF_sim_M)->Timing.stepSize0,
      (&rtmGetErrorStatus((&Planar_robot_3DoF_sim_M))),
      "matlab_torque",
      SS_DOUBLE,
      0,
      0,
      0,
      3,
      1,
      dimensions,
      NO_LOGVALDIMS,
      (nullptr),
      (nullptr),
      0,
      1,
      0.001,
      1);
    if (Planar_robot_3DoF_sim_DW.ToWorkspace1_PWORK.LoggedData == (nullptr))
      return;
  }

  /* SystemInitialize for Atomic SubSystem: '<Root>/planar_robot3dof_FD' */
  /* InitializeConditions for DiscreteIntegrator: '<S1>/Discrete-Time Integrator1' */
  Planar_robot_3DoF_sim_DW.DiscreteTimeIntegrator1_DSTATE[0] =
    Planar_robot_3DoF_sim_B.init_pos[0];
  Planar_robot_3DoF_sim_DW.DiscreteTimeIntegrator1_DSTATE[1] =
    Planar_robot_3DoF_sim_B.init_pos[1];
  Planar_robot_3DoF_sim_DW.DiscreteTimeIntegrator1_DSTATE[2] =
    Planar_robot_3DoF_sim_B.init_pos[2];

  /* End of SystemInitialize for SubSystem: '<Root>/planar_robot3dof_FD' */
}

/* Model terminate function */
void Planar_robot_3DoF_sim::terminate()
{
  /* (no terminate code required) */
}

/* Constructor */
Planar_robot_3DoF_sim::Planar_robot_3DoF_sim() :
  Planar_robot_3DoF_sim_B(),
  Planar_robot_3DoF_sim_DW(),
  Planar_robot_3DoF_sim_M()
{
  /* Currently there is no constructor body generated.*/
}

/* Destructor */
/* Currently there is no destructor body generated.*/
Planar_robot_3DoF_sim::~Planar_robot_3DoF_sim() = default;

/* Real-Time Model get method */
RT_MODEL_Planar_robot_3DoF_si_T * Planar_robot_3DoF_sim::getRTM()
{
  return (&Planar_robot_3DoF_sim_M);
}
