#pragma once
#include "rednose/helpers/common_ekf.h"
extern "C" {
void live_update_4(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_9(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_10(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_12(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_35(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_32(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_13(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_14(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_update_33(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void live_H(double *in_vec, double *out_5864220398347747596);
void live_err_fun(double *nom_x, double *delta_x, double *out_2500085137790340427);
void live_inv_err_fun(double *nom_x, double *true_x, double *out_5850215098972926867);
void live_H_mod_fun(double *state, double *out_8835204987853825536);
void live_f_fun(double *state, double dt, double *out_127607991900174595);
void live_F_fun(double *state, double dt, double *out_5480754559180280437);
void live_h_4(double *state, double *unused, double *out_401284292074783206);
void live_H_4(double *state, double *unused, double *out_3519454863506354843);
void live_h_9(double *state, double *unused, double *out_6469163290663361896);
void live_H_9(double *state, double *unused, double *out_7640070274938749303);
void live_h_10(double *state, double *unused, double *out_3035724408359026182);
void live_H_10(double *state, double *unused, double *out_5367808086823923002);
void live_h_12(double *state, double *unused, double *out_2685219580534383655);
void live_H_12(double *state, double *unused, double *out_8538911271538316638);
void live_h_35(double *state, double *unused, double *out_1040527124124271540);
void live_H_35(double *state, double *unused, double *out_6886116920878962219);
void live_h_32(double *state, double *unused, double *out_5067673893183247419);
void live_H_32(double *state, double *unused, double *out_5759322780622976839);
void live_h_13(double *state, double *unused, double *out_1605415305938632325);
void live_H_13(double *state, double *unused, double *out_863508566724709031);
void live_h_14(double *state, double *unused, double *out_6469163290663361896);
void live_H_14(double *state, double *unused, double *out_7640070274938749303);
void live_h_33(double *state, double *unused, double *out_3835180837527987847);
void live_H_33(double *state, double *unused, double *out_8410070148191731793);
void live_predict(double *in_x, double *in_P, double *in_Q, double dt);
}