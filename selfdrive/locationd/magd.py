#!/usr/bin/env python3
import os
import sys
import signal
import numpy as np

import cereal.messaging as messaging
from cereal import car, log
from common.params import Params
from common.realtime import config_realtime_process
from common.filter_simple import FirstOrderFilter
from openpilot.system.swaglog import cloudlog
from openpilot.selfdrive.locationd.helpers import PointBuckets

POINTS_PER_BUCKET = 2000
MIN_POINTS_TOTAL = 4000
FIT_POINTS_TOTAL = 4000
MIN_VEL = 5  # m/s
FILTER_DECAY = 1.0
FILTER_DT = 0.015
BUCKET_KEYS = np.arange(0, 4, 1)
MIN_BUCKET_POINTS = 1000
NO_OF_BUCKETS = 4

VERSION = 1  # bump this to invalidate old parameter caches


class MagCalibrator:
  def __init__(self, CP):
    self.reset()

    initial_params = {
      'calibrationParams': [],
      'magneticUncalibrated': [],
      'magneticUncalibratedFiltered': [],
      'magneticCalibrated': [],
      'points': []
    }

    # try to restore cached params
    params = Params()
    params_cache = params.get("MagnetometerCarParams")
    magnetometer_cache = params.get("MagnetometerCalibration")
    if params_cache is not None and magnetometer_cache is not None:
      try:
        with log.Event.from_bytes(magnetometer_cache) as log_evt:
          cache_mc = log_evt.MagnetometerCalibration
        with car.CarParams.from_bytes(params_cache) as msg:
          cache_CP = msg
        if self.get_restore_key(cache_CP, cache_mc.version) == self.get_restore_key(CP, VERSION):
          if cache_mc.calibrated:
            initial_params = {
              'calibrationParams': cache_mc.calibrationParams,
            }
          initial_params['points'] = cache_mc.points
          self.filtered_points.load_points(initial_params['points'])
          cloudlog.info("restored magnetometer calibration params from cache")
      except Exception:
        cloudlog.exception("failed to restore cached magnetometer calibration params")
        params.remove("MagnetometerCarParams")
        params.remove("MagnetometerCalibration")

    self.filtered_magnetic = [FirstOrderFilter(0, FILTER_DECAY, FILTER_DT) for _ in range(3)]
    self.filtered_calib = [FirstOrderFilter(0, FILTER_DECAY, FILTER_DT) for _ in range(5)]
    self.vego = 0
    self.yaw = 0
    self.yaw_valid = False

  def get_restore_key(self, CP, version):
    return (CP.carFingerprint, version)

  def reset(self):
    self.point_buckets = PointBuckets(x_bounds=NO_OF_BUCKETS,
                                        min_points=[MIN_BUCKET_POINTS] * NO_OF_BUCKETS,
                                        min_points_total=MIN_POINTS_TOTAL,
                                        points_per_bucket=POINTS_PER_BUCKET,
                                        rowsize=2)

  def get_ellipsoid_rotation(self, coeffs):
    a, b, c = coeffs[:3]
    M = np.array([
        [2 * a, c],
        [c, 2 * b]
    ])
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    axes_lengths = np.sqrt(1 / eigenvalues)
    rotation_angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    return axes_lengths, rotation_angle

  def get_ellipsoid_center(self, coeffs):
    a, b, c, d, e = coeffs[:5]
    num = (c**2 / 4) - (a * b)
    x0 = ((b * d / 2) - (c / 2 * e / 2)) / num
    y0 = ((a * e / 2) - (c / 2 * d / 2)) / num
    return np.array([x0, y0])

  def estimate_calibration_params(self):
    points = self.point_buckets.get_points(FIT_POINTS_TOTAL)
    try:
      x, y = points[:, 0], points[:, 1]
      D = np.vstack((x * x, y * y, x * y, x, y, np.ones_like(x))).T
      _, _, V = np.linalg.svd(D, full_matrices=False)
      coeffs = V[-1] / V[-1, -1]
      x0, y0 = self.get_ellipsoid_center(coeffs)
      (l1, l2), theta = self.get_ellipsoid_rotation(coeffs)
      calibration_params = np.array([x0, y0, l1, l2, theta])
    except np.linalg.LinAlgError as e:
      cloudlog.exception(f"Error computing magnetometer calibration params: {e}")
      calibration_params = np.ones(5) * np.nan
    return calibration_params

  def calibrate_values(x, y, calibration_params):
    x0, y0, l1, l2, theta = calibration_params
    x = x - x0
    y = y - y0
    x_new = x * np.cos(theta) + y * np.sin(theta)
    y_new = -x * np.sin(theta) + y * np.cos(theta)
    return x_new / l1, y_new / l2

  def handle_log(self, which, msg):
    if which == "carState":
      self.vego = msg.carState.vEgo
    elif which == "liveLocationKalman":
      self.yaw = msg.liveLocationKalman.orientationNED.value[2]
      self.yaw_valid = msg.liveLocationKalman.orientationNED.valid
    elif which == "magnetometer":
      raw_vals = np.array(msg.magnetometer.magneticUncalibrated.v)
      self.raw_vals = (raw_vals[: 3] - raw_vals[3:]) / 2
      if np.all(np.abs(self.raw_vals - self.past_raw_vals) < 0.5) and msg.magnetometer.magneticUncalibrated.valid:
        self.filtered_vals = np.array([f.update(v) for f, v in zip(self.filtered_magnetic, self.raw_vals, strict=True)])
      if self.vego > MIN_VEL and self.yaw_valid:
        self.point_buckets.add_point(self.filtered_vals[0], self.filtered_vals[2])
      self.past_raw_vals = self.raw_vals

  def get_msg(self, valid=True, with_points=False):
    msg = messaging.new_message('MagnetometerCalibration')
    msg.valid = valid
    MagnetometerCalibration = msg.MagnetometerCalibration
    MagnetometerCalibration.version = VERSION

    calibration_params = np.ones(5) * np.nan
    filtered_calibration_params = np.ones(5) * np.nan
    calibrated_vals = np.nan * np.ones(2)
    calibrated = False
    bearing = np.nan

    if self.point_buckets.is_valid():
      calibration_params = self.estimate_calibration_params()
      filtered_calibration_params = np.array([f.update(v) for f, v in zip(self.filtered_calib, calibration_params, strict=True)])
      calibrated_vals = self.calibrate_values(self.filtered_vals[0], self.filtered_vals[2], filtered_calibration_params)
      calibrated = True
      bearing = np.arctan2(calibrated_vals[1], calibrated_vals[0]) * 180.0 / np.pi
      bearing = (bearing + 360.0 + 45.0) % 360

    MagnetometerCalibration.calibrated = calibrated
    MagnetometerCalibration.rawVals = self.raw_vals.tolist()
    MagnetometerCalibration.filteredVals = self.filtered_vals.tolist()
    MagnetometerCalibration.calibrationParams = calibration_params.tolist()
    MagnetometerCalibration.filteredCalibrationParams = filtered_calibration_params.tolist()
    MagnetometerCalibration.calibratedVals = calibrated_vals.tolist()
    MagnetometerCalibration.bearing = float(bearing)

    return msg


def main(sm=None, pm=None):
  config_realtime_process([0, 1, 2, 3], 5)

  if sm is None:
    sm = messaging.SubMaster(['magnetometer', 'carState', 'liveLocationKalman'], poll=['liveLocationKalman'])

  if pm is None:
    pm = messaging.PubMaster(['magnetommeterCalbration'])

  params = Params()
  with car.CarParams.from_bytes(params.get("CarParams", block=True)) as CP:
    estimator = MagCalibrator(CP)

  def cache_params(sig, frame):
    signal.signal(sig, signal.SIG_DFL)
    cloudlog.warning("caching torque params")

    params = Params()
    params.put("magnetommeterCalbration", CP.as_builder().to_bytes())

    msg = estimator.get_msg(with_points=True)
    params.put("magnetommeterCalbration", msg.to_bytes())

    sys.exit(0)
  if "REPLAY" not in os.environ:
    signal.signal(signal.SIGINT, cache_params)

  while True:
    sm.update()
    if sm.all_checks():
      for which in sm.updated.keys():
        if sm.updated[which]:
          t = sm.logMonoTime[which] * 1e-9
          estimator.handle_log(t, which, sm[which])

    # 4Hz driven by liveLocationKalman
    if sm.frame % 5 == 0:
      pm.send('magnetommeterCalbration', estimator.get_msg(valid=sm.all_checks()))


if __name__ == "__main__":
  main()
