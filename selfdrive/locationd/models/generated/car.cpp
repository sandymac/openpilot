#include "car.h"

namespace {
#define DIM 9
#define EDIM 9
#define MEDIM 9
typedef void (*Hfun)(double *, double *, double *);

double mass;

void set_mass(double x){ mass = x;}

double rotational_inertia;

void set_rotational_inertia(double x){ rotational_inertia = x;}

double center_to_front;

void set_center_to_front(double x){ center_to_front = x;}

double center_to_rear;

void set_center_to_rear(double x){ center_to_rear = x;}

double stiffness_front;

void set_stiffness_front(double x){ stiffness_front = x;}

double stiffness_rear;

void set_stiffness_rear(double x){ stiffness_rear = x;}
const static double MAHA_THRESH_25 = 3.8414588206941227;
const static double MAHA_THRESH_24 = 5.991464547107981;
const static double MAHA_THRESH_30 = 3.8414588206941227;
const static double MAHA_THRESH_26 = 3.8414588206941227;
const static double MAHA_THRESH_27 = 3.8414588206941227;
const static double MAHA_THRESH_29 = 3.8414588206941227;
const static double MAHA_THRESH_28 = 3.8414588206941227;
const static double MAHA_THRESH_31 = 3.8414588206941227;

/******************************************************************************
 *                       Code generated with SymPy 1.12                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_3163603358833363132) {
   out_3163603358833363132[0] = delta_x[0] + nom_x[0];
   out_3163603358833363132[1] = delta_x[1] + nom_x[1];
   out_3163603358833363132[2] = delta_x[2] + nom_x[2];
   out_3163603358833363132[3] = delta_x[3] + nom_x[3];
   out_3163603358833363132[4] = delta_x[4] + nom_x[4];
   out_3163603358833363132[5] = delta_x[5] + nom_x[5];
   out_3163603358833363132[6] = delta_x[6] + nom_x[6];
   out_3163603358833363132[7] = delta_x[7] + nom_x[7];
   out_3163603358833363132[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_3491614544082927011) {
   out_3491614544082927011[0] = -nom_x[0] + true_x[0];
   out_3491614544082927011[1] = -nom_x[1] + true_x[1];
   out_3491614544082927011[2] = -nom_x[2] + true_x[2];
   out_3491614544082927011[3] = -nom_x[3] + true_x[3];
   out_3491614544082927011[4] = -nom_x[4] + true_x[4];
   out_3491614544082927011[5] = -nom_x[5] + true_x[5];
   out_3491614544082927011[6] = -nom_x[6] + true_x[6];
   out_3491614544082927011[7] = -nom_x[7] + true_x[7];
   out_3491614544082927011[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_6522240562772538499) {
   out_6522240562772538499[0] = 1.0;
   out_6522240562772538499[1] = 0;
   out_6522240562772538499[2] = 0;
   out_6522240562772538499[3] = 0;
   out_6522240562772538499[4] = 0;
   out_6522240562772538499[5] = 0;
   out_6522240562772538499[6] = 0;
   out_6522240562772538499[7] = 0;
   out_6522240562772538499[8] = 0;
   out_6522240562772538499[9] = 0;
   out_6522240562772538499[10] = 1.0;
   out_6522240562772538499[11] = 0;
   out_6522240562772538499[12] = 0;
   out_6522240562772538499[13] = 0;
   out_6522240562772538499[14] = 0;
   out_6522240562772538499[15] = 0;
   out_6522240562772538499[16] = 0;
   out_6522240562772538499[17] = 0;
   out_6522240562772538499[18] = 0;
   out_6522240562772538499[19] = 0;
   out_6522240562772538499[20] = 1.0;
   out_6522240562772538499[21] = 0;
   out_6522240562772538499[22] = 0;
   out_6522240562772538499[23] = 0;
   out_6522240562772538499[24] = 0;
   out_6522240562772538499[25] = 0;
   out_6522240562772538499[26] = 0;
   out_6522240562772538499[27] = 0;
   out_6522240562772538499[28] = 0;
   out_6522240562772538499[29] = 0;
   out_6522240562772538499[30] = 1.0;
   out_6522240562772538499[31] = 0;
   out_6522240562772538499[32] = 0;
   out_6522240562772538499[33] = 0;
   out_6522240562772538499[34] = 0;
   out_6522240562772538499[35] = 0;
   out_6522240562772538499[36] = 0;
   out_6522240562772538499[37] = 0;
   out_6522240562772538499[38] = 0;
   out_6522240562772538499[39] = 0;
   out_6522240562772538499[40] = 1.0;
   out_6522240562772538499[41] = 0;
   out_6522240562772538499[42] = 0;
   out_6522240562772538499[43] = 0;
   out_6522240562772538499[44] = 0;
   out_6522240562772538499[45] = 0;
   out_6522240562772538499[46] = 0;
   out_6522240562772538499[47] = 0;
   out_6522240562772538499[48] = 0;
   out_6522240562772538499[49] = 0;
   out_6522240562772538499[50] = 1.0;
   out_6522240562772538499[51] = 0;
   out_6522240562772538499[52] = 0;
   out_6522240562772538499[53] = 0;
   out_6522240562772538499[54] = 0;
   out_6522240562772538499[55] = 0;
   out_6522240562772538499[56] = 0;
   out_6522240562772538499[57] = 0;
   out_6522240562772538499[58] = 0;
   out_6522240562772538499[59] = 0;
   out_6522240562772538499[60] = 1.0;
   out_6522240562772538499[61] = 0;
   out_6522240562772538499[62] = 0;
   out_6522240562772538499[63] = 0;
   out_6522240562772538499[64] = 0;
   out_6522240562772538499[65] = 0;
   out_6522240562772538499[66] = 0;
   out_6522240562772538499[67] = 0;
   out_6522240562772538499[68] = 0;
   out_6522240562772538499[69] = 0;
   out_6522240562772538499[70] = 1.0;
   out_6522240562772538499[71] = 0;
   out_6522240562772538499[72] = 0;
   out_6522240562772538499[73] = 0;
   out_6522240562772538499[74] = 0;
   out_6522240562772538499[75] = 0;
   out_6522240562772538499[76] = 0;
   out_6522240562772538499[77] = 0;
   out_6522240562772538499[78] = 0;
   out_6522240562772538499[79] = 0;
   out_6522240562772538499[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_3881265194926261560) {
   out_3881265194926261560[0] = state[0];
   out_3881265194926261560[1] = state[1];
   out_3881265194926261560[2] = state[2];
   out_3881265194926261560[3] = state[3];
   out_3881265194926261560[4] = state[4];
   out_3881265194926261560[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8000000000000007*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_3881265194926261560[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_3881265194926261560[7] = state[7];
   out_3881265194926261560[8] = state[8];
}
void F_fun(double *state, double dt, double *out_795201357153754441) {
   out_795201357153754441[0] = 1;
   out_795201357153754441[1] = 0;
   out_795201357153754441[2] = 0;
   out_795201357153754441[3] = 0;
   out_795201357153754441[4] = 0;
   out_795201357153754441[5] = 0;
   out_795201357153754441[6] = 0;
   out_795201357153754441[7] = 0;
   out_795201357153754441[8] = 0;
   out_795201357153754441[9] = 0;
   out_795201357153754441[10] = 1;
   out_795201357153754441[11] = 0;
   out_795201357153754441[12] = 0;
   out_795201357153754441[13] = 0;
   out_795201357153754441[14] = 0;
   out_795201357153754441[15] = 0;
   out_795201357153754441[16] = 0;
   out_795201357153754441[17] = 0;
   out_795201357153754441[18] = 0;
   out_795201357153754441[19] = 0;
   out_795201357153754441[20] = 1;
   out_795201357153754441[21] = 0;
   out_795201357153754441[22] = 0;
   out_795201357153754441[23] = 0;
   out_795201357153754441[24] = 0;
   out_795201357153754441[25] = 0;
   out_795201357153754441[26] = 0;
   out_795201357153754441[27] = 0;
   out_795201357153754441[28] = 0;
   out_795201357153754441[29] = 0;
   out_795201357153754441[30] = 1;
   out_795201357153754441[31] = 0;
   out_795201357153754441[32] = 0;
   out_795201357153754441[33] = 0;
   out_795201357153754441[34] = 0;
   out_795201357153754441[35] = 0;
   out_795201357153754441[36] = 0;
   out_795201357153754441[37] = 0;
   out_795201357153754441[38] = 0;
   out_795201357153754441[39] = 0;
   out_795201357153754441[40] = 1;
   out_795201357153754441[41] = 0;
   out_795201357153754441[42] = 0;
   out_795201357153754441[43] = 0;
   out_795201357153754441[44] = 0;
   out_795201357153754441[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_795201357153754441[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_795201357153754441[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_795201357153754441[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_795201357153754441[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_795201357153754441[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_795201357153754441[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_795201357153754441[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_795201357153754441[53] = -9.8000000000000007*dt;
   out_795201357153754441[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_795201357153754441[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_795201357153754441[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_795201357153754441[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_795201357153754441[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_795201357153754441[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_795201357153754441[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_795201357153754441[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_795201357153754441[62] = 0;
   out_795201357153754441[63] = 0;
   out_795201357153754441[64] = 0;
   out_795201357153754441[65] = 0;
   out_795201357153754441[66] = 0;
   out_795201357153754441[67] = 0;
   out_795201357153754441[68] = 0;
   out_795201357153754441[69] = 0;
   out_795201357153754441[70] = 1;
   out_795201357153754441[71] = 0;
   out_795201357153754441[72] = 0;
   out_795201357153754441[73] = 0;
   out_795201357153754441[74] = 0;
   out_795201357153754441[75] = 0;
   out_795201357153754441[76] = 0;
   out_795201357153754441[77] = 0;
   out_795201357153754441[78] = 0;
   out_795201357153754441[79] = 0;
   out_795201357153754441[80] = 1;
}
void h_25(double *state, double *unused, double *out_2282463371865912665) {
   out_2282463371865912665[0] = state[6];
}
void H_25(double *state, double *unused, double *out_7297697921304868791) {
   out_7297697921304868791[0] = 0;
   out_7297697921304868791[1] = 0;
   out_7297697921304868791[2] = 0;
   out_7297697921304868791[3] = 0;
   out_7297697921304868791[4] = 0;
   out_7297697921304868791[5] = 0;
   out_7297697921304868791[6] = 1;
   out_7297697921304868791[7] = 0;
   out_7297697921304868791[8] = 0;
}
void h_24(double *state, double *unused, double *out_7447174784330914742) {
   out_7447174784330914742[0] = state[4];
   out_7447174784330914742[1] = state[5];
}
void H_24(double *state, double *unused, double *out_8976396553399183259) {
   out_8976396553399183259[0] = 0;
   out_8976396553399183259[1] = 0;
   out_8976396553399183259[2] = 0;
   out_8976396553399183259[3] = 0;
   out_8976396553399183259[4] = 1;
   out_8976396553399183259[5] = 0;
   out_8976396553399183259[6] = 0;
   out_8976396553399183259[7] = 0;
   out_8976396553399183259[8] = 0;
   out_8976396553399183259[9] = 0;
   out_8976396553399183259[10] = 0;
   out_8976396553399183259[11] = 0;
   out_8976396553399183259[12] = 0;
   out_8976396553399183259[13] = 0;
   out_8976396553399183259[14] = 1;
   out_8976396553399183259[15] = 0;
   out_8976396553399183259[16] = 0;
   out_8976396553399183259[17] = 0;
}
void h_30(double *state, double *unused, double *out_2007269309581406776) {
   out_2007269309581406776[0] = state[4];
}
void H_30(double *state, double *unused, double *out_4779364962797620164) {
   out_4779364962797620164[0] = 0;
   out_4779364962797620164[1] = 0;
   out_4779364962797620164[2] = 0;
   out_4779364962797620164[3] = 0;
   out_4779364962797620164[4] = 1;
   out_4779364962797620164[5] = 0;
   out_4779364962797620164[6] = 0;
   out_4779364962797620164[7] = 0;
   out_4779364962797620164[8] = 0;
}
void h_26(double *state, double *unused, double *out_5248874407512044933) {
   out_5248874407512044933[0] = state[7];
}
void H_26(double *state, double *unused, double *out_7407542833530626601) {
   out_7407542833530626601[0] = 0;
   out_7407542833530626601[1] = 0;
   out_7407542833530626601[2] = 0;
   out_7407542833530626601[3] = 0;
   out_7407542833530626601[4] = 0;
   out_7407542833530626601[5] = 0;
   out_7407542833530626601[6] = 0;
   out_7407542833530626601[7] = 1;
   out_7407542833530626601[8] = 0;
}
void h_27(double *state, double *unused, double *out_9161517998050573172) {
   out_9161517998050573172[0] = state[3];
}
void H_27(double *state, double *unused, double *out_4446586510476649716) {
   out_4446586510476649716[0] = 0;
   out_4446586510476649716[1] = 0;
   out_4446586510476649716[2] = 0;
   out_4446586510476649716[3] = 1;
   out_4446586510476649716[4] = 0;
   out_4446586510476649716[5] = 0;
   out_4446586510476649716[6] = 0;
   out_4446586510476649716[7] = 0;
   out_4446586510476649716[8] = 0;
}
void h_29(double *state, double *unused, double *out_1403953424904929259) {
   out_1403953424904929259[0] = state[1];
}
void H_29(double *state, double *unused, double *out_7131581166591466811) {
   out_7131581166591466811[0] = 0;
   out_7131581166591466811[1] = 1;
   out_7131581166591466811[2] = 0;
   out_7131581166591466811[3] = 0;
   out_7131581166591466811[4] = 0;
   out_7131581166591466811[5] = 0;
   out_7131581166591466811[6] = 0;
   out_7131581166591466811[7] = 0;
   out_7131581166591466811[8] = 0;
}
void h_28(double *state, double *unused, double *out_4283552807998132799) {
   out_4283552807998132799[0] = state[0];
}
void H_28(double *state, double *unused, double *out_9095211438156793062) {
   out_9095211438156793062[0] = 1;
   out_9095211438156793062[1] = 0;
   out_9095211438156793062[2] = 0;
   out_9095211438156793062[3] = 0;
   out_9095211438156793062[4] = 0;
   out_9095211438156793062[5] = 0;
   out_9095211438156793062[6] = 0;
   out_9095211438156793062[7] = 0;
   out_9095211438156793062[8] = 0;
}
void h_31(double *state, double *unused, double *out_6523634086499151648) {
   out_6523634086499151648[0] = state[8];
}
void H_31(double *state, double *unused, double *out_6781334731297275125) {
   out_6781334731297275125[0] = 0;
   out_6781334731297275125[1] = 0;
   out_6781334731297275125[2] = 0;
   out_6781334731297275125[3] = 0;
   out_6781334731297275125[4] = 0;
   out_6781334731297275125[5] = 0;
   out_6781334731297275125[6] = 0;
   out_6781334731297275125[7] = 0;
   out_6781334731297275125[8] = 1;
}
#include <eigen3/Eigen/Dense>
#include <iostream>

typedef Eigen::Matrix<double, DIM, DIM, Eigen::RowMajor> DDM;
typedef Eigen::Matrix<double, EDIM, EDIM, Eigen::RowMajor> EEM;
typedef Eigen::Matrix<double, DIM, EDIM, Eigen::RowMajor> DEM;

void predict(double *in_x, double *in_P, double *in_Q, double dt) {
  typedef Eigen::Matrix<double, MEDIM, MEDIM, Eigen::RowMajor> RRM;

  double nx[DIM] = {0};
  double in_F[EDIM*EDIM] = {0};

  // functions from sympy
  f_fun(in_x, dt, nx);
  F_fun(in_x, dt, in_F);


  EEM F(in_F);
  EEM P(in_P);
  EEM Q(in_Q);

  RRM F_main = F.topLeftCorner(MEDIM, MEDIM);
  P.topLeftCorner(MEDIM, MEDIM) = (F_main * P.topLeftCorner(MEDIM, MEDIM)) * F_main.transpose();
  P.topRightCorner(MEDIM, EDIM - MEDIM) = F_main * P.topRightCorner(MEDIM, EDIM - MEDIM);
  P.bottomLeftCorner(EDIM - MEDIM, MEDIM) = P.bottomLeftCorner(EDIM - MEDIM, MEDIM) * F_main.transpose();

  P = P + dt*Q;

  // copy out state
  memcpy(in_x, nx, DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
}

// note: extra_args dim only correct when null space projecting
// otherwise 1
template <int ZDIM, int EADIM, bool MAHA_TEST>
void update(double *in_x, double *in_P, Hfun h_fun, Hfun H_fun, Hfun Hea_fun, double *in_z, double *in_R, double *in_ea, double MAHA_THRESHOLD) {
  typedef Eigen::Matrix<double, ZDIM, ZDIM, Eigen::RowMajor> ZZM;
  typedef Eigen::Matrix<double, ZDIM, DIM, Eigen::RowMajor> ZDM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, EDIM, Eigen::RowMajor> XEM;
  //typedef Eigen::Matrix<double, EDIM, ZDIM, Eigen::RowMajor> EZM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> X1M;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> XXM;

  double in_hx[ZDIM] = {0};
  double in_H[ZDIM * DIM] = {0};
  double in_H_mod[EDIM * DIM] = {0};
  double delta_x[EDIM] = {0};
  double x_new[DIM] = {0};


  // state x, P
  Eigen::Matrix<double, ZDIM, 1> z(in_z);
  EEM P(in_P);
  ZZM pre_R(in_R);

  // functions from sympy
  h_fun(in_x, in_ea, in_hx);
  H_fun(in_x, in_ea, in_H);
  ZDM pre_H(in_H);

  // get y (y = z - hx)
  Eigen::Matrix<double, ZDIM, 1> pre_y(in_hx); pre_y = z - pre_y;
  X1M y; XXM H; XXM R;
  if (Hea_fun){
    typedef Eigen::Matrix<double, ZDIM, EADIM, Eigen::RowMajor> ZAM;
    double in_Hea[ZDIM * EADIM] = {0};
    Hea_fun(in_x, in_ea, in_Hea);
    ZAM Hea(in_Hea);
    XXM A = Hea.transpose().fullPivLu().kernel();


    y = A.transpose() * pre_y;
    H = A.transpose() * pre_H;
    R = A.transpose() * pre_R * A;
  } else {
    y = pre_y;
    H = pre_H;
    R = pre_R;
  }
  // get modified H
  H_mod_fun(in_x, in_H_mod);
  DEM H_mod(in_H_mod);
  XEM H_err = H * H_mod;

  // Do mahalobis distance test
  if (MAHA_TEST){
    XXM a = (H_err * P * H_err.transpose() + R).inverse();
    double maha_dist = y.transpose() * a * y;
    if (maha_dist > MAHA_THRESHOLD){
      R = 1.0e16 * R;
    }
  }

  // Outlier resilient weighting
  double weight = 1;//(1.5)/(1 + y.squaredNorm()/R.sum());

  // kalman gains and I_KH
  XXM S = ((H_err * P) * H_err.transpose()) + R/weight;
  XEM KT = S.fullPivLu().solve(H_err * P.transpose());
  //EZM K = KT.transpose(); TODO: WHY DOES THIS NOT COMPILE?
  //EZM K = S.fullPivLu().solve(H_err * P.transpose()).transpose();
  //std::cout << "Here is the matrix rot:\n" << K << std::endl;
  EEM I_KH = Eigen::Matrix<double, EDIM, EDIM>::Identity() - (KT.transpose() * H_err);

  // update state by injecting dx
  Eigen::Matrix<double, EDIM, 1> dx(delta_x);
  dx  = (KT.transpose() * y);
  memcpy(delta_x, dx.data(), EDIM * sizeof(double));
  err_fun(in_x, delta_x, x_new);
  Eigen::Matrix<double, DIM, 1> x(x_new);

  // update cov
  P = ((I_KH * P) * I_KH.transpose()) + ((KT.transpose() * R) * KT);

  // copy out state
  memcpy(in_x, x.data(), DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
  memcpy(in_z, y.data(), y.rows() * sizeof(double));
}




}
extern "C" {

void car_update_25(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_25, H_25, NULL, in_z, in_R, in_ea, MAHA_THRESH_25);
}
void car_update_24(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<2, 3, 0>(in_x, in_P, h_24, H_24, NULL, in_z, in_R, in_ea, MAHA_THRESH_24);
}
void car_update_30(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_30, H_30, NULL, in_z, in_R, in_ea, MAHA_THRESH_30);
}
void car_update_26(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_26, H_26, NULL, in_z, in_R, in_ea, MAHA_THRESH_26);
}
void car_update_27(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_27, H_27, NULL, in_z, in_R, in_ea, MAHA_THRESH_27);
}
void car_update_29(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_29, H_29, NULL, in_z, in_R, in_ea, MAHA_THRESH_29);
}
void car_update_28(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_28, H_28, NULL, in_z, in_R, in_ea, MAHA_THRESH_28);
}
void car_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_31, H_31, NULL, in_z, in_R, in_ea, MAHA_THRESH_31);
}
void car_err_fun(double *nom_x, double *delta_x, double *out_3163603358833363132) {
  err_fun(nom_x, delta_x, out_3163603358833363132);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_3491614544082927011) {
  inv_err_fun(nom_x, true_x, out_3491614544082927011);
}
void car_H_mod_fun(double *state, double *out_6522240562772538499) {
  H_mod_fun(state, out_6522240562772538499);
}
void car_f_fun(double *state, double dt, double *out_3881265194926261560) {
  f_fun(state,  dt, out_3881265194926261560);
}
void car_F_fun(double *state, double dt, double *out_795201357153754441) {
  F_fun(state,  dt, out_795201357153754441);
}
void car_h_25(double *state, double *unused, double *out_2282463371865912665) {
  h_25(state, unused, out_2282463371865912665);
}
void car_H_25(double *state, double *unused, double *out_7297697921304868791) {
  H_25(state, unused, out_7297697921304868791);
}
void car_h_24(double *state, double *unused, double *out_7447174784330914742) {
  h_24(state, unused, out_7447174784330914742);
}
void car_H_24(double *state, double *unused, double *out_8976396553399183259) {
  H_24(state, unused, out_8976396553399183259);
}
void car_h_30(double *state, double *unused, double *out_2007269309581406776) {
  h_30(state, unused, out_2007269309581406776);
}
void car_H_30(double *state, double *unused, double *out_4779364962797620164) {
  H_30(state, unused, out_4779364962797620164);
}
void car_h_26(double *state, double *unused, double *out_5248874407512044933) {
  h_26(state, unused, out_5248874407512044933);
}
void car_H_26(double *state, double *unused, double *out_7407542833530626601) {
  H_26(state, unused, out_7407542833530626601);
}
void car_h_27(double *state, double *unused, double *out_9161517998050573172) {
  h_27(state, unused, out_9161517998050573172);
}
void car_H_27(double *state, double *unused, double *out_4446586510476649716) {
  H_27(state, unused, out_4446586510476649716);
}
void car_h_29(double *state, double *unused, double *out_1403953424904929259) {
  h_29(state, unused, out_1403953424904929259);
}
void car_H_29(double *state, double *unused, double *out_7131581166591466811) {
  H_29(state, unused, out_7131581166591466811);
}
void car_h_28(double *state, double *unused, double *out_4283552807998132799) {
  h_28(state, unused, out_4283552807998132799);
}
void car_H_28(double *state, double *unused, double *out_9095211438156793062) {
  H_28(state, unused, out_9095211438156793062);
}
void car_h_31(double *state, double *unused, double *out_6523634086499151648) {
  h_31(state, unused, out_6523634086499151648);
}
void car_H_31(double *state, double *unused, double *out_6781334731297275125) {
  H_31(state, unused, out_6781334731297275125);
}
void car_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
}
void car_set_mass(double x) {
  set_mass(x);
}
void car_set_rotational_inertia(double x) {
  set_rotational_inertia(x);
}
void car_set_center_to_front(double x) {
  set_center_to_front(x);
}
void car_set_center_to_rear(double x) {
  set_center_to_rear(x);
}
void car_set_stiffness_front(double x) {
  set_stiffness_front(x);
}
void car_set_stiffness_rear(double x) {
  set_stiffness_rear(x);
}
}

const EKF car = {
  .name = "car",
  .kinds = { 25, 24, 30, 26, 27, 29, 28, 31 },
  .feature_kinds = {  },
  .f_fun = car_f_fun,
  .F_fun = car_F_fun,
  .err_fun = car_err_fun,
  .inv_err_fun = car_inv_err_fun,
  .H_mod_fun = car_H_mod_fun,
  .predict = car_predict,
  .hs = {
    { 25, car_h_25 },
    { 24, car_h_24 },
    { 30, car_h_30 },
    { 26, car_h_26 },
    { 27, car_h_27 },
    { 29, car_h_29 },
    { 28, car_h_28 },
    { 31, car_h_31 },
  },
  .Hs = {
    { 25, car_H_25 },
    { 24, car_H_24 },
    { 30, car_H_30 },
    { 26, car_H_26 },
    { 27, car_H_27 },
    { 29, car_H_29 },
    { 28, car_H_28 },
    { 31, car_H_31 },
  },
  .updates = {
    { 25, car_update_25 },
    { 24, car_update_24 },
    { 30, car_update_30 },
    { 26, car_update_26 },
    { 27, car_update_27 },
    { 29, car_update_29 },
    { 28, car_update_28 },
    { 31, car_update_31 },
  },
  .Hes = {
  },
  .sets = {
    { "mass", car_set_mass },
    { "rotational_inertia", car_set_rotational_inertia },
    { "center_to_front", car_set_center_to_front },
    { "center_to_rear", car_set_center_to_rear },
    { "stiffness_front", car_set_stiffness_front },
    { "stiffness_rear", car_set_stiffness_rear },
  },
  .extra_routines = {
  },
};

ekf_init(car);
