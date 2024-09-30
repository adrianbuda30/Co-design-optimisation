//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: calcSysMatrices.cpp
//
// MATLAB Coder version            : 5.6
// C/C++ source code generated on  : 23-Nov-2023 17:42:08
//

// Include Files
#include "calcSysMatrices.h"
#include "mldivide.h"
#include "mtimes.h"
#include <algorithm>
#include <cmath>
#include <cstring>

// Function Declarations
static void binary_expand_op(double in1[9], int in2, const double in3_data[],
                             const int in3_size[2]);

static void binary_expand_op(double in1[9], int in2, const double in3_data[],
                             const int in3_size[2], const double in4_data[],
                             const int in4_size[2]);

// Function Definitions
//
// Arguments    : double in1[9]
//                int in2
//                const double in3_data[]
//                const int in3_size[2]
// Return Type  : void
//
static void binary_expand_op(double in1[9], int in2, const double in3_data[],
                             const int in3_size[2])
{
  double in1_data[9];
  int aux_0_1;
  int aux_1_1;
  int b_loop_ub;
  int loop_ub;
  int stride_0_0_tmp;
  int stride_1_0;
  int stride_1_1;
  if (in3_size[0] == 1) {
    loop_ub = in2 + 1;
  } else {
    loop_ub = in3_size[0];
  }
  if (in3_size[1] == 1) {
    b_loop_ub = in2 + 1;
  } else {
    b_loop_ub = in3_size[1];
  }
  stride_0_0_tmp = (in2 + 1 != 1);
  stride_1_0 = (in3_size[0] != 1);
  stride_1_1 = (in3_size[1] != 1);
  aux_0_1 = 0;
  aux_1_1 = 0;
  for (int i{0}; i < b_loop_ub; i++) {
    for (int i1{0}; i1 < loop_ub; i1++) {
      in1_data[i1 + loop_ub * i] =
          in1[i1 * stride_0_0_tmp + 3 * aux_0_1] +
          in3_data[i1 * stride_1_0 + in3_size[0] * aux_1_1];
    }
    aux_1_1 += stride_1_1;
    aux_0_1 += stride_0_0_tmp;
  }
  for (int i{0}; i < b_loop_ub; i++) {
    for (int i1{0}; i1 < loop_ub; i1++) {
      in1[i1 + 3 * i] = in1_data[i1 + loop_ub * i];
    }
  }
}

//
// Arguments    : double in1[9]
//                int in2
//                const double in3_data[]
//                const int in3_size[2]
//                const double in4_data[]
//                const int in4_size[2]
// Return Type  : void
//
static void binary_expand_op(double in1[9], int in2, const double in3_data[],
                             const int in3_size[2], const double in4_data[],
                             const int in4_size[2])
{
  double in1_data[9];
  int aux_0_1;
  int aux_1_1;
  int aux_2_1;
  int b_loop_ub;
  int loop_ub;
  int stride_0_0_tmp;
  int stride_1_0;
  int stride_1_1;
  int stride_2_0;
  int stride_2_1;
  if (in4_size[0] == 1) {
    if (in3_size[0] == 1) {
      loop_ub = in2 + 1;
    } else {
      loop_ub = in3_size[0];
    }
  } else {
    loop_ub = in4_size[0];
  }
  if (in4_size[1] == 1) {
    if (in3_size[1] == 1) {
      b_loop_ub = in2 + 1;
    } else {
      b_loop_ub = in3_size[1];
    }
  } else {
    b_loop_ub = in4_size[1];
  }
  stride_0_0_tmp = (in2 + 1 != 1);
  stride_1_0 = (in3_size[0] != 1);
  stride_1_1 = (in3_size[1] != 1);
  stride_2_0 = (in4_size[0] != 1);
  stride_2_1 = (in4_size[1] != 1);
  aux_0_1 = 0;
  aux_1_1 = 0;
  aux_2_1 = 0;
  for (int i{0}; i < b_loop_ub; i++) {
    for (int i1{0}; i1 < loop_ub; i1++) {
      in1_data[i1 + loop_ub * i] =
          (in1[i1 * stride_0_0_tmp + 3 * aux_0_1] +
           in3_data[i1 * stride_1_0 + in3_size[0] * aux_1_1]) +
          in4_data[i1 * stride_2_0 + in4_size[0] * aux_2_1];
    }
    aux_2_1 += stride_2_1;
    aux_1_1 += stride_1_1;
    aux_0_1 += stride_0_0_tmp;
  }
  for (int i{0}; i < b_loop_ub; i++) {
    for (int i1{0}; i1 < loop_ub; i1++) {
      in1[i1 + 3 * i] = in1_data[i1 + loop_ub * i];
    }
  }
}

//
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
// Arguments    : const double q[3]
//                const double dq[3]
//                double rho
//                double radius
//                const double arm_length[3]
//                const double torque[3]
//                double joint_acc[3]
//                double pos_tcp[3]
// Return Type  : void
//
void calcSysMatrices(const double q[3], const double dq[3], double rho,
                     double radius, const double arm_length[3],
                     const double torque[3], double joint_acc[3],
                     double pos_tcp[3])
{
  static const signed char b[9]{1, 0, 0, 0, 1, 0, 0, 0, 1};
  static const signed char a[6]{0, 0, 0, 0, 0, 1};
  double ROBOT_Mass[108];
  double ROBOT_g0[48];
  double J[18];
  double J_pre[18];
  double dJ[18];
  double dJ_pre[18];
  double O_Htm[16];
  double O_Htm_pre[16];
  double b_O_Htm[16];
  double CC[9];
  double M[9];
  double S[9];
  double b_CC[9];
  double b_I[9];
  double b_S[9];
  double b_out_tmp[9];
  double c_I[9];
  double c_S[9];
  double d_I[9];
  double jointOrigins[9];
  double out_tmp[9];
  double tmp2[9];
  double y[9];
  double inertialTwist_pre[6];
  double g[3];
  double Iz;
  double b_m;
  double body_CenterOfMass_idx_0;
  double c_m;
  double d;
  double d1;
  double d2;
  double m;
  double m_tmp;
  double s;
  int C_size[2];
  int J_size[2];
  int tmp_size[2];
  int aoffset;
  int i;
  int i1;
  int ibcol;
  int idxStart_4Row;
  int k;
  signed char ROBOT_csi[18];
  signed char E[9];
  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  //  Calculation of system matrices
  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  //
  //  Calculates the dynamic and kinematic matrices by using the iterative
  //  algorithm proposed in [1]. Kinematic and inertia parameters of the robot
  //  can be given in a compact way. Additionally, this algorithm can ensure
  //  the skew-symmetric property of dM - 2C.
  //
  //  Inputs:
  //  q:        Current joint positions, n-by-1 vector
  //  dq:       Current joint velocities, n-by-1 vector
  //  ROBOT:    Robot parameters, MATLAB structure with three
  //            fields:
  //            g0:       4n-by-4 matrix stacking vertically the homogeneous
  //                      transformation matrices from parent to link in the
  //                      zero-configuration
  //            csi:      6n-by-1 vector stacking vertically the 6-by-1
  //                      relative body jacobian (twist)
  //            Mass:     6n-by-6 matrix stacking vertically the 6-by-6
  //                      constant body inertia matrix from parent to link
  //            g_vec:    3-by-1 gravitational acceleration constant
  //            tcp_t_ee: 4-by-4 homogeneous transformation matrix from
  //                      end-effector to tcp
  //
  //  Outputs
  //  M:        n-by-n joint space inertia matrix
  //  CC:       n-by-n joint space Cor./centrifugal matrix
  //  g:        n-by-1 gravity torque vector
  //  O_J_TCP:  6-by-n hybrid jacobian of tcp
  //  B_J_TCP:  6-by-n body jacobian of tcp
  //  Htm_TCP:  4-by-4 homogeneous transformation matrix from tcp to base
  //
  //  References
  //  [1] Garofalo, Gianluca, Christian Ott and Alin Albu-Schäffer. "On the
  //  closed form computation of the dynamic matrices and their
  //  differentiations." 2013 IEEE/RSJ International Conference on Intelligent
  //  Robots and Systems (2013): 2364-2359.
  //
  //  Implemented by Xuwei Wu.
  //  Planar3DoF Description.
  //    Planar manipulator with 3 DoF.
  //    ...
  //  Origin of the frames and axis
  for (i = 0; i < 9; i++) {
    E[i] = 0;
  }
  //  Initial origins of joint coordinate systems represented in global
  //  coordinate system (one for each column)
  jointOrigins[0] = 0.0;
  jointOrigins[3] = arm_length[0];
  jointOrigins[6] = arm_length[1];
  E[0] = 1;
  jointOrigins[1] = 0.0;
  jointOrigins[2] = 0.0;
  E[4] = 1;
  jointOrigins[4] = 0.0;
  jointOrigins[5] = 0.0;
  E[8] = 1;
  jointOrigins[7] = 0.0;
  jointOrigins[8] = 0.0;
  //  Parameter of end-effector
  //  homogeneous transformation of tcp
  //  Mass, Center of Mass, and Inertia calculations for each link
  //  Compute dynamics based on cylindrical geometry
  s = radius * radius;
  m_tmp = 3.1415926535897931 * s;
  m = m_tmp * arm_length[0] * rho;
  //  Mass
  body_CenterOfMass_idx_0 = arm_length[0] / 2.0;
  //  Center of mass
  //  Inertia around z-axis
  Iz =
      0.25 * m * s + 0.083333333333333329 * m * (arm_length[0] * arm_length[0]);
  //  Inertia around x and y axis
  //  Assign computed values to body
  //  Spatial inertia matrix
  b_m = m;
  std::memset(&b_I[0], 0, 9U * sizeof(double));
  //  Inertia  Spatial rigid-body inertia from mass, CoM, and rotational
  //  inertia.
  //  ...
  //  utility functions -------------------------------------------------------%
  //  Skew symmetric matrix.
  //  ...
  b_I[0] = 0.5 * m * s;
  S[0] = 0.0;
  b_I[4] = Iz;
  S[3] = 0.0;
  b_I[8] = Iz;
  S[6] = 0.0;
  S[1] = 0.0;
  S[4] = 0.0;
  S[7] = -body_CenterOfMass_idx_0;
  S[2] = -0.0;
  S[5] = body_CenterOfMass_idx_0;
  S[8] = 0.0;
  for (i = 0; i < 9; i++) {
    tmp2[i] = m * S[i];
    M[i] = m * static_cast<double>(b[i]);
  }
  //  Initial relative homogeneous transformation matrix
  //  Compute dynamics based on cylindrical geometry
  m = m_tmp * arm_length[1] * rho;
  //  Mass
  body_CenterOfMass_idx_0 = arm_length[1] / 2.0;
  //  Center of mass
  //  Inertia around z-axis
  Iz =
      0.25 * m * s + 0.083333333333333329 * m * (arm_length[1] * arm_length[1]);
  //  Inertia around x and y axis
  //  Assign computed values to body
  //  Spatial inertia matrix
  c_m = m;
  std::memset(&CC[0], 0, 9U * sizeof(double));
  //  Inertia  Spatial rigid-body inertia from mass, CoM, and rotational
  //  inertia.
  //  ...
  //  utility functions -------------------------------------------------------%
  //  Skew symmetric matrix.
  //  ...
  CC[0] = 0.5 * m * s;
  b_S[0] = 0.0;
  CC[4] = Iz;
  b_S[3] = 0.0;
  CC[8] = Iz;
  b_S[6] = 0.0;
  b_S[1] = 0.0;
  b_S[4] = 0.0;
  b_S[7] = -body_CenterOfMass_idx_0;
  b_S[2] = -0.0;
  b_S[5] = body_CenterOfMass_idx_0;
  b_S[8] = 0.0;
  for (i = 0; i < 9; i++) {
    out_tmp[i] = m * b_S[i];
    y[i] = m * static_cast<double>(b[i]);
  }
  //  Initial relative homogeneous transformation matrix
  //  Compute dynamics based on cylindrical geometry
  m = m_tmp * arm_length[2] * rho;
  //  Mass
  body_CenterOfMass_idx_0 = arm_length[2] / 2.0;
  //  Center of mass
  //  Inertia around z-axis
  Iz =
      0.25 * m * s + 0.083333333333333329 * m * (arm_length[2] * arm_length[2]);
  //  Inertia around x and y axis
  //  Assign computed values to body
  //  Spatial inertia matrix
  std::memset(&c_I[0], 0, 9U * sizeof(double));
  //  Inertia  Spatial rigid-body inertia from mass, CoM, and rotational
  //  inertia.
  //  ...
  //  utility functions -------------------------------------------------------%
  //  Skew symmetric matrix.
  //  ...
  c_I[0] = 0.5 * m * s;
  c_S[0] = 0.0;
  c_I[4] = Iz;
  c_S[3] = 0.0;
  c_I[8] = Iz;
  c_S[6] = 0.0;
  c_S[1] = 0.0;
  c_S[4] = 0.0;
  c_S[7] = -body_CenterOfMass_idx_0;
  c_S[2] = -0.0;
  c_S[5] = body_CenterOfMass_idx_0;
  c_S[8] = 0.0;
  for (i = 0; i < 9; i++) {
    b_out_tmp[i] = m * c_S[i];
  }
  //  Initial relative homogeneous transformation matrix
  //  Relative body jacobians adjusted for 3 DOF planar robot (rotation around
  //  z-axis) Gravitational acceleration constant Pack into ROBOT structure
  //  ROBOT.g0 = cell2mat(g0);
  //  ROBOT.Mass = cell2mat(Mass);
  for (aoffset = 0; aoffset < 3; aoffset++) {
    ibcol = aoffset * 6;
    for (k = 0; k < 6; k++) {
      ROBOT_csi[ibcol + k] = a[k];
    }
    ROBOT_g0[12 * aoffset] = E[3 * aoffset];
    ROBOT_g0[12 * aoffset + 1] = E[3 * aoffset + 1];
    ROBOT_g0[12 * aoffset + 2] = E[3 * aoffset + 2];
    ROBOT_g0[aoffset + 36] = 0.0;
  }
  ROBOT_g0[3] = 0.0;
  ROBOT_g0[15] = 0.0;
  ROBOT_g0[27] = 0.0;
  ROBOT_g0[39] = 1.0;
  for (i = 0; i < 3; i++) {
    ROBOT_g0[12 * i + 4] = E[3 * i];
    ROBOT_g0[12 * i + 5] = E[3 * i + 1];
    ROBOT_g0[12 * i + 6] = E[3 * i + 2];
    ROBOT_g0[i + 40] = jointOrigins[i + 3];
  }
  ROBOT_g0[7] = 0.0;
  ROBOT_g0[19] = 0.0;
  ROBOT_g0[31] = 0.0;
  ROBOT_g0[43] = 1.0;
  for (i = 0; i < 3; i++) {
    ROBOT_g0[12 * i + 8] = E[3 * i];
    ROBOT_g0[12 * i + 9] = E[3 * i + 1];
    ROBOT_g0[12 * i + 10] = E[3 * i + 2];
    ROBOT_g0[i + 44] = jointOrigins[i + 6];
  }
  ROBOT_g0[11] = 0.0;
  ROBOT_g0[23] = 0.0;
  ROBOT_g0[35] = 0.0;
  ROBOT_g0[47] = 1.0;
  for (i = 0; i < 3; i++) {
    d = out_tmp[i];
    d1 = out_tmp[i + 3];
    d2 = out_tmp[i + 6];
    Iz = b_out_tmp[i];
    m_tmp = b_out_tmp[i + 3];
    body_CenterOfMass_idx_0 = b_out_tmp[i + 6];
    for (i1 = 0; i1 < 3; i1++) {
      idxStart_4Row = 3 * i1 + 1;
      aoffset = 3 * i1 + 2;
      ibcol = i + 3 * i1;
      d_I[ibcol] =
          b_I[ibcol] - ((tmp2[i] * S[3 * i1] + tmp2[i + 3] * S[idxStart_4Row]) +
                        tmp2[i + 6] * S[aoffset]);
      b_CC[ibcol] = CC[ibcol] - ((d * b_S[3 * i1] + d1 * b_S[idxStart_4Row]) +
                                 d2 * b_S[aoffset]);
      b_I[ibcol] =
          c_I[ibcol] - ((Iz * c_S[3 * i1] + m_tmp * c_S[idxStart_4Row]) +
                        body_CenterOfMass_idx_0 * c_S[aoffset]);
      aoffset = i1 + 3 * i;
      ibcol = i1 + 18 * i;
      ROBOT_Mass[ibcol] = M[aoffset];
      ROBOT_Mass[i1 + 18 * (i + 3)] = -b_m * S[aoffset];
      ROBOT_Mass[ibcol + 3] = tmp2[aoffset];
    }
  }
  for (i = 0; i < 3; i++) {
    aoffset = 18 * (i + 3);
    ROBOT_Mass[aoffset + 3] = d_I[3 * i];
    ROBOT_Mass[18 * i + 6] = y[3 * i];
    ROBOT_Mass[aoffset + 6] = -c_m * b_S[3 * i];
    ROBOT_Mass[18 * i + 9] = out_tmp[3 * i];
    ROBOT_Mass[aoffset + 9] = b_CC[3 * i];
    ROBOT_Mass[18 * i + 12] = m * static_cast<double>(b[3 * i]);
    ROBOT_Mass[aoffset + 12] = -m * c_S[3 * i];
    ROBOT_Mass[18 * i + 15] = b_out_tmp[3 * i];
    ROBOT_Mass[aoffset + 15] = b_I[3 * i];
    ibcol = 3 * i + 1;
    ROBOT_Mass[aoffset + 4] = d_I[ibcol];
    ROBOT_Mass[18 * i + 7] = y[ibcol];
    ROBOT_Mass[aoffset + 7] = -c_m * b_S[ibcol];
    ROBOT_Mass[18 * i + 10] = out_tmp[ibcol];
    ROBOT_Mass[aoffset + 10] = b_CC[ibcol];
    ROBOT_Mass[18 * i + 13] = m * static_cast<double>(b[ibcol]);
    ROBOT_Mass[aoffset + 13] = -m * c_S[ibcol];
    ROBOT_Mass[18 * i + 16] = b_out_tmp[ibcol];
    ROBOT_Mass[aoffset + 16] = b_I[ibcol];
    ibcol = 3 * i + 2;
    ROBOT_Mass[aoffset + 5] = d_I[ibcol];
    ROBOT_Mass[18 * i + 8] = y[ibcol];
    ROBOT_Mass[aoffset + 8] = -c_m * b_S[ibcol];
    ROBOT_Mass[18 * i + 11] = out_tmp[ibcol];
    ROBOT_Mass[aoffset + 11] = b_CC[ibcol];
    ROBOT_Mass[18 * i + 14] = m * static_cast<double>(b[ibcol]);
    ROBOT_Mass[aoffset + 14] = -m * c_S[ibcol];
    ROBOT_Mass[18 * i + 17] = b_out_tmp[ibcol];
    ROBOT_Mass[aoffset + 17] = b_I[ibcol];
  }
  //  reshape signals
  //  pre-allocate memory for variables
  std::memset(&M[0], 0, 9U * sizeof(double));
  std::memset(&CC[0], 0, 9U * sizeof(double));
  g[0] = 0.0;
  g[1] = 0.0;
  g[2] = 0.0;
  std::memset(&J_pre[0], 0, 18U * sizeof(double));
  std::memset(&dJ_pre[0], 0, 18U * sizeof(double));
  for (int b_i{0}; b_i < 6; b_i++) {
    inertialTwist_pre[b_i] = 0.0;
  }
  std::memset(&O_Htm_pre[0], 0, 16U * sizeof(double));
  O_Htm_pre[0] = 1.0;
  O_Htm_pre[5] = 1.0;
  O_Htm_pre[10] = 1.0;
  O_Htm_pre[15] = 1.0;
  //  iterate from the root to the end-effector of the serial manipulator
  d_I[0] = 0.0;
  d_I[4] = 0.0;
  d_I[8] = 0.0;
  tmp_size[1] = 6;
  C_size[1] = 6;
  d = dq[0];
  d1 = dq[1];
  d2 = dq[2];
  for (int iLink{0}; iLink < 3; iLink++) {
    double Htm_data[48];
    double invAd[36];
    double C_data[18];
    double inertialTwist[6];
    double relTwist[6];
    double c_CC[3];
    int M_size[2];
    int b_tmp_size[2];
    int coffset;
    int idxStart_6Row_tmp;
    signed char b_x_tmp_tmp;
    signed char x_tmp;
    signed char x_tmp_tmp;
    boolean_T x[3];
    boolean_T b_y;
    boolean_T exitg1;
    //  initialize indices of the current iteration
    idxStart_4Row = iLink << 2;
    idxStart_6Row_tmp = 6 * iLink;
    //  compute relative body twist
    for (i = 0; i < 6; i++) {
      relTwist[i] =
          static_cast<double>(ROBOT_csi[idxStart_6Row_tmp + i]) * dq[iLink];
    }
    //  compute relative homogeneous transformation matrix between link and
    //  its parent
    //  Twist exponential of the relative body jacobian
    //  Implemented by Gianluca Garofalo.
    std::memset(&b_I[0], 0, 9U * sizeof(double));
    b_I[0] = 1.0;
    x_tmp_tmp = ROBOT_csi[idxStart_6Row_tmp + 3];
    x[0] = (x_tmp_tmp == 0);
    b_I[4] = 1.0;
    x_tmp = ROBOT_csi[idxStart_6Row_tmp + 4];
    x[1] = (x_tmp == 0);
    b_I[8] = 1.0;
    b_x_tmp_tmp = ROBOT_csi[idxStart_6Row_tmp + 5];
    x[2] = (b_x_tmp_tmp == 0);
    b_y = true;
    k = 0;
    exitg1 = false;
    while ((!exitg1) && (k < 3)) {
      if (!x[k]) {
        b_y = false;
        exitg1 = true;
      } else {
        k++;
      }
    }
    if (b_y) {
      for (i = 0; i < 3; i++) {
        aoffset = i << 2;
        O_Htm[aoffset] = b_I[3 * i];
        O_Htm[aoffset + 1] = b_I[3 * i + 1];
        O_Htm[aoffset + 2] = b_I[3 * i + 2];
        O_Htm[i + 12] =
            static_cast<double>(ROBOT_csi[idxStart_6Row_tmp + i]) * q[iLink];
      }
      O_Htm[3] = 0.0;
      O_Htm[7] = 0.0;
      O_Htm[11] = 0.0;
      O_Htm[15] = 1.0;
    } else {
      //  Skew operator
      //  Implemented by Gianluca Garofalo.
      jointOrigins[0] = 0.0;
      jointOrigins[3] = -static_cast<double>(b_x_tmp_tmp);
      jointOrigins[6] = x_tmp;
      jointOrigins[1] = b_x_tmp_tmp;
      jointOrigins[4] = 0.0;
      jointOrigins[7] = -static_cast<double>(x_tmp_tmp);
      jointOrigins[2] = -static_cast<double>(x_tmp);
      jointOrigins[5] = x_tmp_tmp;
      jointOrigins[8] = 0.0;
      Iz = q[iLink];
      m_tmp = std::sin(Iz);
      s = std::cos(Iz);
      for (i = 0; i < 3; i++) {
        for (i1 = 0; i1 < 3; i1++) {
          S[i + 3 * i1] = (jointOrigins[i] * jointOrigins[3 * i1] +
                           jointOrigins[i + 3] * jointOrigins[3 * i1 + 1]) +
                          jointOrigins[i + 6] * jointOrigins[3 * i1 + 2];
        }
      }
      for (i = 0; i < 3; i++) {
        aoffset = i << 2;
        O_Htm[aoffset] =
            b_I[3 * i] + (jointOrigins[3 * i] * m_tmp + S[3 * i] * (1.0 - s));
        ibcol = 3 * i + 1;
        O_Htm[aoffset + 1] =
            b_I[ibcol] + (jointOrigins[ibcol] * m_tmp + S[ibcol] * (1.0 - s));
        ibcol = 3 * i + 2;
        O_Htm[aoffset + 2] =
            b_I[ibcol] + (jointOrigins[ibcol] * m_tmp + S[ibcol] * (1.0 - s));
        O_Htm[i + 12] = 0.0;
      }
      O_Htm[3] = 0.0;
      O_Htm[7] = 0.0;
      O_Htm[11] = 0.0;
      O_Htm[15] = 1.0;
    }
    for (int j{0}; j < 4; j++) {
      coffset = j * 4;
      ibcol = j << 2;
      Iz = O_Htm[ibcol];
      m_tmp = O_Htm[ibcol + 1];
      body_CenterOfMass_idx_0 = O_Htm[ibcol + 2];
      s = O_Htm[ibcol + 3];
      for (int b_i{0}; b_i < 4; b_i++) {
        Htm_data[coffset + b_i] =
            ((ROBOT_g0[idxStart_4Row + b_i % 4] * Iz +
              ROBOT_g0[(idxStart_4Row + (4 + b_i) % 4) + 12] * m_tmp) +
             ROBOT_g0[(idxStart_4Row + (8 + b_i) % 4) + 24] *
                 body_CenterOfMass_idx_0) +
            ROBOT_g0[(idxStart_4Row + (12 + b_i) % 4) + 36] * s;
      }
    }
    //  propagate rotation matrix, body twists, body jacobians and
    //  derivatives of body jacobians
    for (i = 0; i < 4; i++) {
      Iz = O_Htm_pre[i];
      m_tmp = O_Htm_pre[i + 4];
      body_CenterOfMass_idx_0 = O_Htm_pre[i + 8];
      s = O_Htm_pre[i + 12];
      for (i1 = 0; i1 < 4; i1++) {
        idxStart_4Row = i1 << 2;
        O_Htm[i + idxStart_4Row] =
            ((Iz * Htm_data[idxStart_4Row] +
              m_tmp * Htm_data[idxStart_4Row + 1]) +
             body_CenterOfMass_idx_0 * Htm_data[idxStart_4Row + 2]) +
            s * Htm_data[idxStart_4Row + 3];
      }
    }
    for (int b_i{0}; b_i < 6; b_i++) {
      inertialTwist[b_i] = relTwist[b_i];
    }
    //  Inverse of the adjoint transformation
    //  Implemented by Gianluca Garofalo.
    for (i = 0; i < 3; i++) {
      jointOrigins[3 * i] = Htm_data[i];
      jointOrigins[3 * i + 1] = Htm_data[i + 4];
      jointOrigins[3 * i + 2] = Htm_data[i + 8];
    }
    //  Skew operator
    //  Implemented by Gianluca Garofalo.
    for (i = 0; i < 9; i++) {
      S[i] = -jointOrigins[i];
    }
    d_I[3] = -Htm_data[14];
    d_I[6] = Htm_data[13];
    d_I[1] = Htm_data[14];
    d_I[7] = -Htm_data[12];
    d_I[2] = -Htm_data[13];
    d_I[5] = Htm_data[12];
    for (i = 0; i < 3; i++) {
      Iz = S[i];
      m_tmp = S[i + 3];
      body_CenterOfMass_idx_0 = S[i + 6];
      for (i1 = 0; i1 < 3; i1++) {
        b_CC[i + 3 * i1] = (Iz * d_I[3 * i1] + m_tmp * d_I[3 * i1 + 1]) +
                           body_CenterOfMass_idx_0 * d_I[3 * i1 + 2];
        invAd[i1 + 6 * i] = jointOrigins[i1 + 3 * i];
      }
    }
    for (i = 0; i < 3; i++) {
      aoffset = 6 * (i + 3);
      invAd[aoffset] = b_CC[3 * i];
      invAd[6 * i + 3] = 0.0;
      invAd[aoffset + 3] = jointOrigins[3 * i];
      ibcol = 3 * i + 1;
      invAd[aoffset + 1] = b_CC[ibcol];
      invAd[6 * i + 4] = 0.0;
      invAd[aoffset + 4] = jointOrigins[ibcol];
      ibcol = 3 * i + 2;
      invAd[aoffset + 2] = b_CC[ibcol];
      invAd[6 * i + 5] = 0.0;
      invAd[aoffset + 5] = jointOrigins[ibcol];
    }
    std::memset(&J[0], 0, 18U * sizeof(double));
    std::memset(&dJ[0], 0, 18U * sizeof(double));
    if (iLink + 1 > 1) {
      double B_data[12];
      double b_C_data[12];
      double c_C_data[12];
      for (i = 0; i < 6; i++) {
        Iz = 0.0;
        for (i1 = 0; i1 < 6; i1++) {
          Iz += invAd[i + 6 * i1] * inertialTwist_pre[i1];
        }
        inertialTwist[i] = relTwist[i] + Iz;
      }
      for (i = 0; i < iLink; i++) {
        for (i1 = 0; i1 < 6; i1++) {
          aoffset = i1 + 6 * i;
          B_data[aoffset] = J_pre[aoffset];
        }
      }
      for (int j{0}; j < iLink; j++) {
        aoffset = j * 6;
        for (int b_i{0}; b_i < 6; b_i++) {
          s = 0.0;
          for (k = 0; k < 6; k++) {
            s += invAd[k * 6 + b_i] * B_data[aoffset + k];
          }
          b_C_data[aoffset + b_i] = s;
        }
      }
      //  Lie bracket of the body twist
      //  Implemented by Gianluca Garofalo.
      //  Skew operator
      //  Implemented by Gianluca Garofalo.
      //  Skew operator
      //  Implemented by Gianluca Garofalo.
      jointOrigins[0] = 0.0;
      jointOrigins[3] = -relTwist[5];
      jointOrigins[6] = relTwist[4];
      jointOrigins[1] = relTwist[5];
      jointOrigins[4] = 0.0;
      jointOrigins[7] = -relTwist[3];
      jointOrigins[2] = -relTwist[4];
      jointOrigins[5] = relTwist[3];
      jointOrigins[8] = 0.0;
      for (i = 0; i < iLink; i++) {
        for (i1 = 0; i1 < 6; i1++) {
          aoffset = i1 + 6 * i;
          J[aoffset] = b_C_data[aoffset];
          B_data[aoffset] = dJ_pre[aoffset];
        }
      }
      for (int j{0}; j < iLink; j++) {
        aoffset = j * 6;
        for (int b_i{0}; b_i < 6; b_i++) {
          s = 0.0;
          for (k = 0; k < 6; k++) {
            s += invAd[k * 6 + b_i] * B_data[aoffset + k];
          }
          b_C_data[aoffset + b_i] = s;
        }
      }
      invAd[18] = 0.0;
      invAd[24] = -relTwist[2];
      invAd[30] = relTwist[1];
      invAd[19] = relTwist[2];
      invAd[25] = 0.0;
      invAd[31] = -relTwist[0];
      invAd[20] = -relTwist[1];
      invAd[26] = relTwist[0];
      invAd[32] = 0.0;
      for (i = 0; i < 3; i++) {
        Iz = jointOrigins[3 * i];
        invAd[6 * i] = Iz;
        invAd[6 * i + 3] = 0.0;
        aoffset = 6 * (i + 3);
        invAd[aoffset + 3] = Iz;
        Iz = jointOrigins[3 * i + 1];
        invAd[6 * i + 1] = Iz;
        invAd[6 * i + 4] = 0.0;
        invAd[aoffset + 4] = Iz;
        Iz = jointOrigins[3 * i + 2];
        invAd[6 * i + 2] = Iz;
        invAd[6 * i + 5] = 0.0;
        invAd[aoffset + 5] = Iz;
      }
      for (i = 0; i < iLink; i++) {
        for (i1 = 0; i1 < 6; i1++) {
          aoffset = i1 + 6 * i;
          B_data[aoffset] = J[aoffset];
        }
      }
      for (int j{0}; j < iLink; j++) {
        aoffset = j * 6;
        for (int b_i{0}; b_i < 6; b_i++) {
          s = 0.0;
          for (k = 0; k < 6; k++) {
            s += invAd[k * 6 + b_i] * B_data[aoffset + k];
          }
          c_C_data[aoffset + b_i] = s;
        }
      }
      for (i = 0; i < iLink; i++) {
        for (i1 = 0; i1 < 6; i1++) {
          aoffset = i1 + 6 * i;
          dJ[aoffset] = b_C_data[aoffset] - c_C_data[aoffset];
        }
      }
    }
    for (i = 0; i < 6; i++) {
      aoffset = idxStart_6Row_tmp + i;
      J[aoffset] = ROBOT_csi[aoffset];
    }
    //  update inertia matrix
    tmp_size[0] = iLink + 1;
    for (int j{0}; j < 6; j++) {
      coffset = j * (iLink + 1);
      ibcol = j * 6;
      for (int b_i{0}; b_i <= iLink; b_i++) {
        aoffset = b_i * 6;
        s = 0.0;
        for (k = 0; k < 6; k++) {
          i = aoffset + k;
          i1 = ibcol + k;
          s += J[i % 6 + 6 * (i / 6)] *
               ROBOT_Mass[(idxStart_6Row_tmp + i1 % 6) + 18 * (i1 / 6)];
        }
        J_pre[coffset + b_i] = s;
      }
    }
    J_size[0] = 6;
    J_size[1] = iLink + 1;
    for (i = 0; i <= iLink; i++) {
      for (i1 = 0; i1 < 6; i1++) {
        aoffset = i1 + 6 * i;
        dJ_pre[aoffset] = J[aoffset];
      }
    }
    coder::internal::blas::mtimes(J_pre, tmp_size, dJ_pre, J_size, jointOrigins,
                                  M_size);
    if ((iLink + 1 == M_size[0]) && (iLink + 1 == M_size[1])) {
      for (i = 0; i <= iLink; i++) {
        for (i1 = 0; i1 <= iLink; i1++) {
          aoffset = i1 + (iLink + 1) * i;
          jointOrigins[aoffset] += M[i1 + 3 * i];
        }
      }
      aoffset = iLink + 1;
      ibcol = iLink + 1;
      for (i = 0; i < aoffset; i++) {
        for (i1 = 0; i1 < ibcol; i1++) {
          M[i1 + 3 * i] = jointOrigins[i1 + (iLink + 1) * i];
        }
      }
    } else {
      binary_expand_op(M, iLink, jointOrigins, M_size);
    }
    //  update Cor.\centrifugal matrix
    //  Method 1: Original version from [1]
    //  Mass_x_ad_tmp = ROBOT.Mass(idxStart_6Row:idxEnd_6Row, 1:6) *
    //  LieBracket(inertialTwist); Mass_x_ad_tmp = Mass_x_ad_tmp -
    //  Mass_x_ad_tmp'; Skew operator Implemented by Gianluca Garofalo.
    jointOrigins[0] = 0.0;
    jointOrigins[3] = -inertialTwist[5];
    jointOrigins[6] = inertialTwist[4];
    jointOrigins[1] = inertialTwist[5];
    jointOrigins[4] = 0.0;
    jointOrigins[7] = -inertialTwist[3];
    jointOrigins[2] = -inertialTwist[4];
    jointOrigins[5] = inertialTwist[3];
    jointOrigins[8] = 0.0;
    for (i = 0; i < 3; i++) {
      i1 = (idxStart_6Row_tmp + i) + 3;
      for (idxStart_4Row = 0; idxStart_4Row < 3; idxStart_4Row++) {
        Iz = jointOrigins[3 * idxStart_4Row];
        m_tmp = ROBOT_Mass[i1] * Iz;
        body_CenterOfMass_idx_0 = ROBOT_Mass[i1 + 54] * Iz;
        Iz = jointOrigins[3 * idxStart_4Row + 1];
        m_tmp += ROBOT_Mass[i1 + 18] * Iz;
        body_CenterOfMass_idx_0 += ROBOT_Mass[i1 + 72] * Iz;
        Iz = jointOrigins[3 * idxStart_4Row + 2];
        m_tmp += ROBOT_Mass[i1 + 36] * Iz;
        body_CenterOfMass_idx_0 += ROBOT_Mass[i1 + 90] * Iz;
        aoffset = i + 3 * idxStart_4Row;
        tmp2[aoffset] = body_CenterOfMass_idx_0;
        S[aoffset] = m_tmp;
        invAd[idxStart_4Row + 6 * i] =
            ROBOT_Mass[idxStart_6Row_tmp] * jointOrigins[idxStart_4Row + 3 * i];
        invAd[idxStart_4Row + 6 * (i + 3)] = -m_tmp;
      }
    }
    for (i = 0; i < 3; i++) {
      invAd[6 * i + 3] = S[3 * i];
      aoffset = 6 * (i + 3);
      invAd[aoffset + 3] = tmp2[3 * i] - tmp2[i];
      ibcol = 3 * i + 1;
      invAd[6 * i + 4] = S[ibcol];
      invAd[aoffset + 4] = tmp2[ibcol] - tmp2[i + 3];
      ibcol = 3 * i + 2;
      invAd[6 * i + 5] = S[ibcol];
      invAd[aoffset + 5] = tmp2[ibcol] - tmp2[i + 6];
    }
    C_size[0] = iLink + 1;
    for (int j{0}; j < 6; j++) {
      coffset = j * (iLink + 1);
      ibcol = j * 6;
      for (int b_i{0}; b_i <= iLink; b_i++) {
        aoffset = b_i * 6;
        s = 0.0;
        for (k = 0; k < 6; k++) {
          i = aoffset + k;
          s += J[i % 6 + 6 * (i / 6)] * invAd[ibcol + k];
        }
        C_data[coffset + b_i] = s;
      }
    }
    J_size[0] = 6;
    J_size[1] = iLink + 1;
    for (i = 0; i <= iLink; i++) {
      for (i1 = 0; i1 < 6; i1++) {
        aoffset = i1 + 6 * i;
        dJ_pre[aoffset] = J[aoffset];
      }
    }
    coder::internal::blas::mtimes(C_data, C_size, dJ_pre, J_size, jointOrigins,
                                  M_size);
    J_size[0] = 6;
    J_size[1] = iLink + 1;
    for (i = 0; i <= iLink; i++) {
      for (i1 = 0; i1 < 6; i1++) {
        aoffset = i1 + 6 * i;
        dJ_pre[aoffset] = dJ[aoffset];
      }
    }
    coder::internal::blas::mtimes(J_pre, tmp_size, dJ_pre, J_size, b_I,
                                  b_tmp_size);
    if (iLink + 1 == 1) {
      i = M_size[0];
      i1 = M_size[1];
    } else {
      i = iLink + 1;
      i1 = iLink + 1;
    }
    if ((iLink + 1 == M_size[0]) && (iLink + 1 == M_size[1]) &&
        (i == b_tmp_size[0]) && (i1 == b_tmp_size[1])) {
      for (i = 0; i <= iLink; i++) {
        for (i1 = 0; i1 <= iLink; i1++) {
          aoffset = i1 + (iLink + 1) * i;
          jointOrigins[aoffset] = (CC[i1 + 3 * i] + jointOrigins[aoffset]) +
                                  b_I[i1 + b_tmp_size[0] * i];
        }
      }
      aoffset = iLink + 1;
      ibcol = iLink + 1;
      for (i = 0; i < aoffset; i++) {
        for (i1 = 0; i1 < ibcol; i1++) {
          CC[i1 + 3 * i] = jointOrigins[i1 + (iLink + 1) * i];
        }
      }
    } else {
      binary_expand_op(CC, iLink, jointOrigins, M_size, b_I, b_tmp_size);
    }
    //  update gravity torque vector
    aoffset = iLink + 1;
    for (int j{0}; j < 3; j++) {
      coffset = j * (iLink + 1);
      for (int b_i{0}; b_i <= iLink; b_i++) {
        i = (iLink + b_i) + 1;
        s = J_pre[b_i % (iLink + 1) + (iLink + 1) * (b_i / (iLink + 1))] *
                O_Htm[j % 3] +
            J_pre[i % (iLink + 1) + (iLink + 1) * (i / (iLink + 1))] *
                O_Htm[(j + 3) % 3 + 4];
        i = ((iLink + 1) << 1) + b_i;
        s += J_pre[i % (iLink + 1) + (iLink + 1) * (i / (iLink + 1))] *
             O_Htm[(j + 6) % 3 + 8];
        jointOrigins[coffset + b_i] = s;
      }
    }
    for (int b_i{0}; b_i < aoffset; b_i++) {
      c_CC[b_i] =
          (jointOrigins[b_i] * 0.0 + jointOrigins[(iLink + b_i) + 1] * 0.0) +
          jointOrigins[((iLink + 1) << 1) + b_i] * -9.81;
    }
    aoffset = iLink + 1;
    for (i = 0; i < aoffset; i++) {
      g[i] += c_CC[i];
    }
    //  update variables of parent link
    std::copy(&J[0], &J[18], &J_pre[0]);
    std::copy(&dJ[0], &dJ[18], &dJ_pre[0]);
    for (int b_i{0}; b_i < 6; b_i++) {
      inertialTwist_pre[b_i] = inertialTwist[b_i];
    }
    std::copy(&O_Htm[0], &O_Htm[16], &O_Htm_pre[0]);
    for (i = 0; i < 9; i++) {
      b_CC[i] = -CC[i];
    }
    for (i = 0; i < 3; i++) {
      c_CC[i] =
          (((b_CC[i] * d + b_CC[i + 3] * d1) + b_CC[i + 6] * d2) + torque[i]) -
          g[i];
    }
    coder::mldivide(M, c_CC, joint_acc);
  }
  for (i = 0; i < 3; i++) {
    aoffset = i << 2;
    O_Htm_pre[aoffset] = E[3 * i];
    O_Htm_pre[aoffset + 1] = E[3 * i + 1];
    O_Htm_pre[aoffset + 2] = E[3 * i + 2];
  }
  O_Htm_pre[12] = arm_length[2];
  O_Htm_pre[13] = 0.0;
  O_Htm_pre[14] = 0.0;
  O_Htm_pre[3] = 0.0;
  O_Htm_pre[7] = 0.0;
  O_Htm_pre[11] = 0.0;
  O_Htm_pre[15] = 1.0;
  for (i = 0; i < 4; i++) {
    d = O_Htm[i];
    d1 = O_Htm[i + 4];
    d2 = O_Htm[i + 8];
    Iz = O_Htm[i + 12];
    for (i1 = 0; i1 < 4; i1++) {
      idxStart_4Row = i1 << 2;
      b_O_Htm[i + idxStart_4Row] =
          ((d * O_Htm_pre[idxStart_4Row] + d1 * O_Htm_pre[idxStart_4Row + 1]) +
           d2 * O_Htm_pre[idxStart_4Row + 2]) +
          Iz * O_Htm_pre[idxStart_4Row + 3];
    }
  }
  pos_tcp[0] = b_O_Htm[12];
  pos_tcp[1] = b_O_Htm[13];
  pos_tcp[2] = b_O_Htm[14];
}

//
// File trailer for calcSysMatrices.cpp
//
// [EOF]
//
