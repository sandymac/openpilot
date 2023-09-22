#pragma once
#include "rednose/helpers/common_ekf.h"
extern "C" {
void gnss_update_6(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void gnss_update_20(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void gnss_update_7(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void gnss_update_21(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void gnss_err_fun(double *nom_x, double *delta_x, double *out_737967100360307087);
void gnss_inv_err_fun(double *nom_x, double *true_x, double *out_149885366105645914);
void gnss_H_mod_fun(double *state, double *out_1688414484582863618);
void gnss_f_fun(double *state, double dt, double *out_1211317883374050698);
void gnss_F_fun(double *state, double dt, double *out_5848945878851552988);
void gnss_h_6(double *state, double *sat_pos, double *out_1561008275398760795);
void gnss_H_6(double *state, double *sat_pos, double *out_4996775953226592659);
void gnss_h_20(double *state, double *sat_pos, double *out_2566409088777598106);
void gnss_H_20(double *state, double *sat_pos, double *out_3932547391831848515);
void gnss_h_7(double *state, double *sat_pos_vel, double *out_3844173755684688568);
void gnss_H_7(double *state, double *sat_pos_vel, double *out_1700534350566703411);
void gnss_h_21(double *state, double *sat_pos_vel, double *out_3844173755684688568);
void gnss_H_21(double *state, double *sat_pos_vel, double *out_1700534350566703411);
void gnss_predict(double *in_x, double *in_P, double *in_Q, double dt);
}