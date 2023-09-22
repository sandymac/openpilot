#include "gnss.h"

namespace {
#define DIM 11
#define EDIM 11
#define MEDIM 11
typedef void (*Hfun)(double *, double *, double *);
const static double MAHA_THRESH_6 = 3.8414588206941227;
const static double MAHA_THRESH_20 = 3.8414588206941227;
const static double MAHA_THRESH_7 = 3.8414588206941227;
const static double MAHA_THRESH_21 = 3.8414588206941227;

/******************************************************************************
 *                       Code generated with SymPy 1.12                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_737967100360307087) {
   out_737967100360307087[0] = delta_x[0] + nom_x[0];
   out_737967100360307087[1] = delta_x[1] + nom_x[1];
   out_737967100360307087[2] = delta_x[2] + nom_x[2];
   out_737967100360307087[3] = delta_x[3] + nom_x[3];
   out_737967100360307087[4] = delta_x[4] + nom_x[4];
   out_737967100360307087[5] = delta_x[5] + nom_x[5];
   out_737967100360307087[6] = delta_x[6] + nom_x[6];
   out_737967100360307087[7] = delta_x[7] + nom_x[7];
   out_737967100360307087[8] = delta_x[8] + nom_x[8];
   out_737967100360307087[9] = delta_x[9] + nom_x[9];
   out_737967100360307087[10] = delta_x[10] + nom_x[10];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_149885366105645914) {
   out_149885366105645914[0] = -nom_x[0] + true_x[0];
   out_149885366105645914[1] = -nom_x[1] + true_x[1];
   out_149885366105645914[2] = -nom_x[2] + true_x[2];
   out_149885366105645914[3] = -nom_x[3] + true_x[3];
   out_149885366105645914[4] = -nom_x[4] + true_x[4];
   out_149885366105645914[5] = -nom_x[5] + true_x[5];
   out_149885366105645914[6] = -nom_x[6] + true_x[6];
   out_149885366105645914[7] = -nom_x[7] + true_x[7];
   out_149885366105645914[8] = -nom_x[8] + true_x[8];
   out_149885366105645914[9] = -nom_x[9] + true_x[9];
   out_149885366105645914[10] = -nom_x[10] + true_x[10];
}
void H_mod_fun(double *state, double *out_1688414484582863618) {
   out_1688414484582863618[0] = 1.0;
   out_1688414484582863618[1] = 0;
   out_1688414484582863618[2] = 0;
   out_1688414484582863618[3] = 0;
   out_1688414484582863618[4] = 0;
   out_1688414484582863618[5] = 0;
   out_1688414484582863618[6] = 0;
   out_1688414484582863618[7] = 0;
   out_1688414484582863618[8] = 0;
   out_1688414484582863618[9] = 0;
   out_1688414484582863618[10] = 0;
   out_1688414484582863618[11] = 0;
   out_1688414484582863618[12] = 1.0;
   out_1688414484582863618[13] = 0;
   out_1688414484582863618[14] = 0;
   out_1688414484582863618[15] = 0;
   out_1688414484582863618[16] = 0;
   out_1688414484582863618[17] = 0;
   out_1688414484582863618[18] = 0;
   out_1688414484582863618[19] = 0;
   out_1688414484582863618[20] = 0;
   out_1688414484582863618[21] = 0;
   out_1688414484582863618[22] = 0;
   out_1688414484582863618[23] = 0;
   out_1688414484582863618[24] = 1.0;
   out_1688414484582863618[25] = 0;
   out_1688414484582863618[26] = 0;
   out_1688414484582863618[27] = 0;
   out_1688414484582863618[28] = 0;
   out_1688414484582863618[29] = 0;
   out_1688414484582863618[30] = 0;
   out_1688414484582863618[31] = 0;
   out_1688414484582863618[32] = 0;
   out_1688414484582863618[33] = 0;
   out_1688414484582863618[34] = 0;
   out_1688414484582863618[35] = 0;
   out_1688414484582863618[36] = 1.0;
   out_1688414484582863618[37] = 0;
   out_1688414484582863618[38] = 0;
   out_1688414484582863618[39] = 0;
   out_1688414484582863618[40] = 0;
   out_1688414484582863618[41] = 0;
   out_1688414484582863618[42] = 0;
   out_1688414484582863618[43] = 0;
   out_1688414484582863618[44] = 0;
   out_1688414484582863618[45] = 0;
   out_1688414484582863618[46] = 0;
   out_1688414484582863618[47] = 0;
   out_1688414484582863618[48] = 1.0;
   out_1688414484582863618[49] = 0;
   out_1688414484582863618[50] = 0;
   out_1688414484582863618[51] = 0;
   out_1688414484582863618[52] = 0;
   out_1688414484582863618[53] = 0;
   out_1688414484582863618[54] = 0;
   out_1688414484582863618[55] = 0;
   out_1688414484582863618[56] = 0;
   out_1688414484582863618[57] = 0;
   out_1688414484582863618[58] = 0;
   out_1688414484582863618[59] = 0;
   out_1688414484582863618[60] = 1.0;
   out_1688414484582863618[61] = 0;
   out_1688414484582863618[62] = 0;
   out_1688414484582863618[63] = 0;
   out_1688414484582863618[64] = 0;
   out_1688414484582863618[65] = 0;
   out_1688414484582863618[66] = 0;
   out_1688414484582863618[67] = 0;
   out_1688414484582863618[68] = 0;
   out_1688414484582863618[69] = 0;
   out_1688414484582863618[70] = 0;
   out_1688414484582863618[71] = 0;
   out_1688414484582863618[72] = 1.0;
   out_1688414484582863618[73] = 0;
   out_1688414484582863618[74] = 0;
   out_1688414484582863618[75] = 0;
   out_1688414484582863618[76] = 0;
   out_1688414484582863618[77] = 0;
   out_1688414484582863618[78] = 0;
   out_1688414484582863618[79] = 0;
   out_1688414484582863618[80] = 0;
   out_1688414484582863618[81] = 0;
   out_1688414484582863618[82] = 0;
   out_1688414484582863618[83] = 0;
   out_1688414484582863618[84] = 1.0;
   out_1688414484582863618[85] = 0;
   out_1688414484582863618[86] = 0;
   out_1688414484582863618[87] = 0;
   out_1688414484582863618[88] = 0;
   out_1688414484582863618[89] = 0;
   out_1688414484582863618[90] = 0;
   out_1688414484582863618[91] = 0;
   out_1688414484582863618[92] = 0;
   out_1688414484582863618[93] = 0;
   out_1688414484582863618[94] = 0;
   out_1688414484582863618[95] = 0;
   out_1688414484582863618[96] = 1.0;
   out_1688414484582863618[97] = 0;
   out_1688414484582863618[98] = 0;
   out_1688414484582863618[99] = 0;
   out_1688414484582863618[100] = 0;
   out_1688414484582863618[101] = 0;
   out_1688414484582863618[102] = 0;
   out_1688414484582863618[103] = 0;
   out_1688414484582863618[104] = 0;
   out_1688414484582863618[105] = 0;
   out_1688414484582863618[106] = 0;
   out_1688414484582863618[107] = 0;
   out_1688414484582863618[108] = 1.0;
   out_1688414484582863618[109] = 0;
   out_1688414484582863618[110] = 0;
   out_1688414484582863618[111] = 0;
   out_1688414484582863618[112] = 0;
   out_1688414484582863618[113] = 0;
   out_1688414484582863618[114] = 0;
   out_1688414484582863618[115] = 0;
   out_1688414484582863618[116] = 0;
   out_1688414484582863618[117] = 0;
   out_1688414484582863618[118] = 0;
   out_1688414484582863618[119] = 0;
   out_1688414484582863618[120] = 1.0;
}
void f_fun(double *state, double dt, double *out_1211317883374050698) {
   out_1211317883374050698[0] = dt*state[3] + state[0];
   out_1211317883374050698[1] = dt*state[4] + state[1];
   out_1211317883374050698[2] = dt*state[5] + state[2];
   out_1211317883374050698[3] = state[3];
   out_1211317883374050698[4] = state[4];
   out_1211317883374050698[5] = state[5];
   out_1211317883374050698[6] = dt*state[7] + state[6];
   out_1211317883374050698[7] = dt*state[8] + state[7];
   out_1211317883374050698[8] = state[8];
   out_1211317883374050698[9] = state[9];
   out_1211317883374050698[10] = state[10];
}
void F_fun(double *state, double dt, double *out_5848945878851552988) {
   out_5848945878851552988[0] = 1;
   out_5848945878851552988[1] = 0;
   out_5848945878851552988[2] = 0;
   out_5848945878851552988[3] = dt;
   out_5848945878851552988[4] = 0;
   out_5848945878851552988[5] = 0;
   out_5848945878851552988[6] = 0;
   out_5848945878851552988[7] = 0;
   out_5848945878851552988[8] = 0;
   out_5848945878851552988[9] = 0;
   out_5848945878851552988[10] = 0;
   out_5848945878851552988[11] = 0;
   out_5848945878851552988[12] = 1;
   out_5848945878851552988[13] = 0;
   out_5848945878851552988[14] = 0;
   out_5848945878851552988[15] = dt;
   out_5848945878851552988[16] = 0;
   out_5848945878851552988[17] = 0;
   out_5848945878851552988[18] = 0;
   out_5848945878851552988[19] = 0;
   out_5848945878851552988[20] = 0;
   out_5848945878851552988[21] = 0;
   out_5848945878851552988[22] = 0;
   out_5848945878851552988[23] = 0;
   out_5848945878851552988[24] = 1;
   out_5848945878851552988[25] = 0;
   out_5848945878851552988[26] = 0;
   out_5848945878851552988[27] = dt;
   out_5848945878851552988[28] = 0;
   out_5848945878851552988[29] = 0;
   out_5848945878851552988[30] = 0;
   out_5848945878851552988[31] = 0;
   out_5848945878851552988[32] = 0;
   out_5848945878851552988[33] = 0;
   out_5848945878851552988[34] = 0;
   out_5848945878851552988[35] = 0;
   out_5848945878851552988[36] = 1;
   out_5848945878851552988[37] = 0;
   out_5848945878851552988[38] = 0;
   out_5848945878851552988[39] = 0;
   out_5848945878851552988[40] = 0;
   out_5848945878851552988[41] = 0;
   out_5848945878851552988[42] = 0;
   out_5848945878851552988[43] = 0;
   out_5848945878851552988[44] = 0;
   out_5848945878851552988[45] = 0;
   out_5848945878851552988[46] = 0;
   out_5848945878851552988[47] = 0;
   out_5848945878851552988[48] = 1;
   out_5848945878851552988[49] = 0;
   out_5848945878851552988[50] = 0;
   out_5848945878851552988[51] = 0;
   out_5848945878851552988[52] = 0;
   out_5848945878851552988[53] = 0;
   out_5848945878851552988[54] = 0;
   out_5848945878851552988[55] = 0;
   out_5848945878851552988[56] = 0;
   out_5848945878851552988[57] = 0;
   out_5848945878851552988[58] = 0;
   out_5848945878851552988[59] = 0;
   out_5848945878851552988[60] = 1;
   out_5848945878851552988[61] = 0;
   out_5848945878851552988[62] = 0;
   out_5848945878851552988[63] = 0;
   out_5848945878851552988[64] = 0;
   out_5848945878851552988[65] = 0;
   out_5848945878851552988[66] = 0;
   out_5848945878851552988[67] = 0;
   out_5848945878851552988[68] = 0;
   out_5848945878851552988[69] = 0;
   out_5848945878851552988[70] = 0;
   out_5848945878851552988[71] = 0;
   out_5848945878851552988[72] = 1;
   out_5848945878851552988[73] = dt;
   out_5848945878851552988[74] = 0;
   out_5848945878851552988[75] = 0;
   out_5848945878851552988[76] = 0;
   out_5848945878851552988[77] = 0;
   out_5848945878851552988[78] = 0;
   out_5848945878851552988[79] = 0;
   out_5848945878851552988[80] = 0;
   out_5848945878851552988[81] = 0;
   out_5848945878851552988[82] = 0;
   out_5848945878851552988[83] = 0;
   out_5848945878851552988[84] = 1;
   out_5848945878851552988[85] = dt;
   out_5848945878851552988[86] = 0;
   out_5848945878851552988[87] = 0;
   out_5848945878851552988[88] = 0;
   out_5848945878851552988[89] = 0;
   out_5848945878851552988[90] = 0;
   out_5848945878851552988[91] = 0;
   out_5848945878851552988[92] = 0;
   out_5848945878851552988[93] = 0;
   out_5848945878851552988[94] = 0;
   out_5848945878851552988[95] = 0;
   out_5848945878851552988[96] = 1;
   out_5848945878851552988[97] = 0;
   out_5848945878851552988[98] = 0;
   out_5848945878851552988[99] = 0;
   out_5848945878851552988[100] = 0;
   out_5848945878851552988[101] = 0;
   out_5848945878851552988[102] = 0;
   out_5848945878851552988[103] = 0;
   out_5848945878851552988[104] = 0;
   out_5848945878851552988[105] = 0;
   out_5848945878851552988[106] = 0;
   out_5848945878851552988[107] = 0;
   out_5848945878851552988[108] = 1;
   out_5848945878851552988[109] = 0;
   out_5848945878851552988[110] = 0;
   out_5848945878851552988[111] = 0;
   out_5848945878851552988[112] = 0;
   out_5848945878851552988[113] = 0;
   out_5848945878851552988[114] = 0;
   out_5848945878851552988[115] = 0;
   out_5848945878851552988[116] = 0;
   out_5848945878851552988[117] = 0;
   out_5848945878851552988[118] = 0;
   out_5848945878851552988[119] = 0;
   out_5848945878851552988[120] = 1;
}
void h_6(double *state, double *sat_pos, double *out_1561008275398760795) {
   out_1561008275398760795[0] = sqrt(pow(-sat_pos[0] + state[0], 2) + pow(-sat_pos[1] + state[1], 2) + pow(-sat_pos[2] + state[2], 2)) + state[6];
}
void H_6(double *state, double *sat_pos, double *out_4996775953226592659) {
   out_4996775953226592659[0] = (-sat_pos[0] + state[0])/sqrt(pow(-sat_pos[0] + state[0], 2) + pow(-sat_pos[1] + state[1], 2) + pow(-sat_pos[2] + state[2], 2));
   out_4996775953226592659[1] = (-sat_pos[1] + state[1])/sqrt(pow(-sat_pos[0] + state[0], 2) + pow(-sat_pos[1] + state[1], 2) + pow(-sat_pos[2] + state[2], 2));
   out_4996775953226592659[2] = (-sat_pos[2] + state[2])/sqrt(pow(-sat_pos[0] + state[0], 2) + pow(-sat_pos[1] + state[1], 2) + pow(-sat_pos[2] + state[2], 2));
   out_4996775953226592659[3] = 0;
   out_4996775953226592659[4] = 0;
   out_4996775953226592659[5] = 0;
   out_4996775953226592659[6] = 1;
   out_4996775953226592659[7] = 0;
   out_4996775953226592659[8] = 0;
   out_4996775953226592659[9] = 0;
   out_4996775953226592659[10] = 0;
}
void h_20(double *state, double *sat_pos, double *out_2566409088777598106) {
   out_2566409088777598106[0] = sqrt(pow(-sat_pos[0] + state[0], 2) + pow(-sat_pos[1] + state[1], 2) + pow(-sat_pos[2] + state[2], 2)) + sat_pos[3]*state[10] + state[6] + state[9];
}
void H_20(double *state, double *sat_pos, double *out_3932547391831848515) {
   out_3932547391831848515[0] = (-sat_pos[0] + state[0])/sqrt(pow(-sat_pos[0] + state[0], 2) + pow(-sat_pos[1] + state[1], 2) + pow(-sat_pos[2] + state[2], 2));
   out_3932547391831848515[1] = (-sat_pos[1] + state[1])/sqrt(pow(-sat_pos[0] + state[0], 2) + pow(-sat_pos[1] + state[1], 2) + pow(-sat_pos[2] + state[2], 2));
   out_3932547391831848515[2] = (-sat_pos[2] + state[2])/sqrt(pow(-sat_pos[0] + state[0], 2) + pow(-sat_pos[1] + state[1], 2) + pow(-sat_pos[2] + state[2], 2));
   out_3932547391831848515[3] = 0;
   out_3932547391831848515[4] = 0;
   out_3932547391831848515[5] = 0;
   out_3932547391831848515[6] = 1;
   out_3932547391831848515[7] = 0;
   out_3932547391831848515[8] = 0;
   out_3932547391831848515[9] = 1;
   out_3932547391831848515[10] = sat_pos[3];
}
void h_7(double *state, double *sat_pos_vel, double *out_3844173755684688568) {
   out_3844173755684688568[0] = (sat_pos_vel[0] - state[0])*(sat_pos_vel[3] - state[3])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2)) + (sat_pos_vel[1] - state[1])*(sat_pos_vel[4] - state[4])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2)) + (sat_pos_vel[2] - state[2])*(sat_pos_vel[5] - state[5])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2)) + state[7];
}
void H_7(double *state, double *sat_pos_vel, double *out_1700534350566703411) {
   out_1700534350566703411[0] = pow(sat_pos_vel[0] - state[0], 2)*(sat_pos_vel[3] - state[3])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + (sat_pos_vel[0] - state[0])*(sat_pos_vel[1] - state[1])*(sat_pos_vel[4] - state[4])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + (sat_pos_vel[0] - state[0])*(sat_pos_vel[2] - state[2])*(sat_pos_vel[5] - state[5])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) - (sat_pos_vel[3] - state[3])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_1700534350566703411[1] = (sat_pos_vel[0] - state[0])*(sat_pos_vel[1] - state[1])*(sat_pos_vel[3] - state[3])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + pow(sat_pos_vel[1] - state[1], 2)*(sat_pos_vel[4] - state[4])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + (sat_pos_vel[1] - state[1])*(sat_pos_vel[2] - state[2])*(sat_pos_vel[5] - state[5])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) - (sat_pos_vel[4] - state[4])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_1700534350566703411[2] = (sat_pos_vel[0] - state[0])*(sat_pos_vel[2] - state[2])*(sat_pos_vel[3] - state[3])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + (sat_pos_vel[1] - state[1])*(sat_pos_vel[2] - state[2])*(sat_pos_vel[4] - state[4])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + pow(sat_pos_vel[2] - state[2], 2)*(sat_pos_vel[5] - state[5])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) - (sat_pos_vel[5] - state[5])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_1700534350566703411[3] = -(sat_pos_vel[0] - state[0])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_1700534350566703411[4] = -(sat_pos_vel[1] - state[1])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_1700534350566703411[5] = -(sat_pos_vel[2] - state[2])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_1700534350566703411[6] = 0;
   out_1700534350566703411[7] = 1;
   out_1700534350566703411[8] = 0;
   out_1700534350566703411[9] = 0;
   out_1700534350566703411[10] = 0;
}
void h_21(double *state, double *sat_pos_vel, double *out_3844173755684688568) {
   out_3844173755684688568[0] = (sat_pos_vel[0] - state[0])*(sat_pos_vel[3] - state[3])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2)) + (sat_pos_vel[1] - state[1])*(sat_pos_vel[4] - state[4])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2)) + (sat_pos_vel[2] - state[2])*(sat_pos_vel[5] - state[5])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2)) + state[7];
}
void H_21(double *state, double *sat_pos_vel, double *out_1700534350566703411) {
   out_1700534350566703411[0] = pow(sat_pos_vel[0] - state[0], 2)*(sat_pos_vel[3] - state[3])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + (sat_pos_vel[0] - state[0])*(sat_pos_vel[1] - state[1])*(sat_pos_vel[4] - state[4])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + (sat_pos_vel[0] - state[0])*(sat_pos_vel[2] - state[2])*(sat_pos_vel[5] - state[5])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) - (sat_pos_vel[3] - state[3])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_1700534350566703411[1] = (sat_pos_vel[0] - state[0])*(sat_pos_vel[1] - state[1])*(sat_pos_vel[3] - state[3])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + pow(sat_pos_vel[1] - state[1], 2)*(sat_pos_vel[4] - state[4])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + (sat_pos_vel[1] - state[1])*(sat_pos_vel[2] - state[2])*(sat_pos_vel[5] - state[5])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) - (sat_pos_vel[4] - state[4])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_1700534350566703411[2] = (sat_pos_vel[0] - state[0])*(sat_pos_vel[2] - state[2])*(sat_pos_vel[3] - state[3])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + (sat_pos_vel[1] - state[1])*(sat_pos_vel[2] - state[2])*(sat_pos_vel[4] - state[4])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) + pow(sat_pos_vel[2] - state[2], 2)*(sat_pos_vel[5] - state[5])/pow(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2), 3.0/2.0) - (sat_pos_vel[5] - state[5])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_1700534350566703411[3] = -(sat_pos_vel[0] - state[0])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_1700534350566703411[4] = -(sat_pos_vel[1] - state[1])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_1700534350566703411[5] = -(sat_pos_vel[2] - state[2])/sqrt(pow(sat_pos_vel[0] - state[0], 2) + pow(sat_pos_vel[1] - state[1], 2) + pow(sat_pos_vel[2] - state[2], 2));
   out_1700534350566703411[6] = 0;
   out_1700534350566703411[7] = 1;
   out_1700534350566703411[8] = 0;
   out_1700534350566703411[9] = 0;
   out_1700534350566703411[10] = 0;
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

void gnss_update_6(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_6, H_6, NULL, in_z, in_R, in_ea, MAHA_THRESH_6);
}
void gnss_update_20(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_20, H_20, NULL, in_z, in_R, in_ea, MAHA_THRESH_20);
}
void gnss_update_7(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_7, H_7, NULL, in_z, in_R, in_ea, MAHA_THRESH_7);
}
void gnss_update_21(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_21, H_21, NULL, in_z, in_R, in_ea, MAHA_THRESH_21);
}
void gnss_err_fun(double *nom_x, double *delta_x, double *out_737967100360307087) {
  err_fun(nom_x, delta_x, out_737967100360307087);
}
void gnss_inv_err_fun(double *nom_x, double *true_x, double *out_149885366105645914) {
  inv_err_fun(nom_x, true_x, out_149885366105645914);
}
void gnss_H_mod_fun(double *state, double *out_1688414484582863618) {
  H_mod_fun(state, out_1688414484582863618);
}
void gnss_f_fun(double *state, double dt, double *out_1211317883374050698) {
  f_fun(state,  dt, out_1211317883374050698);
}
void gnss_F_fun(double *state, double dt, double *out_5848945878851552988) {
  F_fun(state,  dt, out_5848945878851552988);
}
void gnss_h_6(double *state, double *sat_pos, double *out_1561008275398760795) {
  h_6(state, sat_pos, out_1561008275398760795);
}
void gnss_H_6(double *state, double *sat_pos, double *out_4996775953226592659) {
  H_6(state, sat_pos, out_4996775953226592659);
}
void gnss_h_20(double *state, double *sat_pos, double *out_2566409088777598106) {
  h_20(state, sat_pos, out_2566409088777598106);
}
void gnss_H_20(double *state, double *sat_pos, double *out_3932547391831848515) {
  H_20(state, sat_pos, out_3932547391831848515);
}
void gnss_h_7(double *state, double *sat_pos_vel, double *out_3844173755684688568) {
  h_7(state, sat_pos_vel, out_3844173755684688568);
}
void gnss_H_7(double *state, double *sat_pos_vel, double *out_1700534350566703411) {
  H_7(state, sat_pos_vel, out_1700534350566703411);
}
void gnss_h_21(double *state, double *sat_pos_vel, double *out_3844173755684688568) {
  h_21(state, sat_pos_vel, out_3844173755684688568);
}
void gnss_H_21(double *state, double *sat_pos_vel, double *out_1700534350566703411) {
  H_21(state, sat_pos_vel, out_1700534350566703411);
}
void gnss_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
}
}

const EKF gnss = {
  .name = "gnss",
  .kinds = { 6, 20, 7, 21 },
  .feature_kinds = {  },
  .f_fun = gnss_f_fun,
  .F_fun = gnss_F_fun,
  .err_fun = gnss_err_fun,
  .inv_err_fun = gnss_inv_err_fun,
  .H_mod_fun = gnss_H_mod_fun,
  .predict = gnss_predict,
  .hs = {
    { 6, gnss_h_6 },
    { 20, gnss_h_20 },
    { 7, gnss_h_7 },
    { 21, gnss_h_21 },
  },
  .Hs = {
    { 6, gnss_H_6 },
    { 20, gnss_H_20 },
    { 7, gnss_H_7 },
    { 21, gnss_H_21 },
  },
  .updates = {
    { 6, gnss_update_6 },
    { 20, gnss_update_20 },
    { 7, gnss_update_7 },
    { 21, gnss_update_21 },
  },
  .Hes = {
  },
  .sets = {
  },
  .extra_routines = {
  },
};

ekf_init(gnss);
