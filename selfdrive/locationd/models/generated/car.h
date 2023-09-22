#pragma once
#include "rednose/helpers/common_ekf.h"
extern "C" {
void car_update_25(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_24(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_30(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_26(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_27(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_29(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_28(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea);
void car_err_fun(double *nom_x, double *delta_x, double *out_3163603358833363132);
void car_inv_err_fun(double *nom_x, double *true_x, double *out_3491614544082927011);
void car_H_mod_fun(double *state, double *out_6522240562772538499);
void car_f_fun(double *state, double dt, double *out_3881265194926261560);
void car_F_fun(double *state, double dt, double *out_795201357153754441);
void car_h_25(double *state, double *unused, double *out_2282463371865912665);
void car_H_25(double *state, double *unused, double *out_7297697921304868791);
void car_h_24(double *state, double *unused, double *out_7447174784330914742);
void car_H_24(double *state, double *unused, double *out_8976396553399183259);
void car_h_30(double *state, double *unused, double *out_2007269309581406776);
void car_H_30(double *state, double *unused, double *out_4779364962797620164);
void car_h_26(double *state, double *unused, double *out_5248874407512044933);
void car_H_26(double *state, double *unused, double *out_7407542833530626601);
void car_h_27(double *state, double *unused, double *out_9161517998050573172);
void car_H_27(double *state, double *unused, double *out_4446586510476649716);
void car_h_29(double *state, double *unused, double *out_1403953424904929259);
void car_H_29(double *state, double *unused, double *out_7131581166591466811);
void car_h_28(double *state, double *unused, double *out_4283552807998132799);
void car_H_28(double *state, double *unused, double *out_9095211438156793062);
void car_h_31(double *state, double *unused, double *out_6523634086499151648);
void car_H_31(double *state, double *unused, double *out_6781334731297275125);
void car_predict(double *in_x, double *in_P, double *in_Q, double dt);
void car_set_mass(double x);
void car_set_rotational_inertia(double x);
void car_set_center_to_front(double x);
void car_set_center_to_rear(double x);
void car_set_stiffness_front(double x);
void car_set_stiffness_rear(double x);
}