import cv2
import numpy as np

from sklearn import linear_model


def average(x, w):
  ret = np.zeros_like(x)

  for i in range(x.shape[0]):
    idx1 = max(0, i - (w - 1) // 2)
    idx2 = min(x.shape[0], i + (w - 1) // 2 + (2 - (w % 2)))

    ret[i] = np.mean(x[idx1:idx2])

  return ret


class calCSpeed:
  def __init__(self, frame_cnt, output_file=None, scale=None, fs_offset=(0, 0)):
    assert output_file is not None or scale is not None

    self.output_file = np.array(output_file)
    self.scale = scale

    self.raw_preds = np.zeros(frame_cnt)

    self.frame_prev = None
    self.pts_prev = None

    self.frame_id = 0


  def extract_features(self, img, mask=None):
    return cv2.goodFeaturesToTrack(img, 30, 0.1, 10, blockSize=10, mask=mask)

  def get_flow(self, frame, frame_prev, pts_prev):
    params = dict( winSize  = (21, 21),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))

    pts, st, _ = cv2.calcOpticalFlowPyrLK(frame_prev, frame, pts_prev, None, **params)

    return np.hstack((pts_prev.reshape(-1, 2), (pts-pts_prev).reshape(-1, 2)))

  def proc_frame(self, img, mask=None):
    img = cv2.GaussianBlur(img, (3, 3), 0)

    if self.pts_prev is None:
      self.raw_preds[self.frame_id] = 0
    else:
      flow = self.get_flow(img, self.frame_prev, self.pts_prev)

      preds = []
      for x, y, u, v in flow:
        if v < -0.05:
          continue

        x -= img.shape[1]/2
        y -= img.shape[0]/2

        if y == 0 or (abs(u) - abs(v)) > 11:
          preds.append(0)
          preds.append(0)
        elif x == 0:
          preds.append(0)
          preds.append(v / (y*y))
        else:
          preds.append(u / (x * y))
          preds.append(v / (y*y))

      preds = [n for n in preds if n >= 0]
      self.raw_preds[self.frame_id] = np.median(preds) if len(preds) else 0
      
    self.pts_prev = self.extract_features(img, mask)
    self.frame_prev = img
    self.frame_id += 1

  # final predictions
  def get_preds(self):
    preds = self.raw_preds * self.get_scale()
    return average(preds, 80)

  def get_scale(self):

    if self.scale is None:
      preds = average(self.raw_preds, 80)
      reg = linear_model.LinearRegression(fit_intercept=False)
      reg.fit(preds.reshape(-1, 1), self.output_file) 
      return reg.coef_[0]

    return self.scale

  def get_frames(self, offset_x=0, offset_y=0):
    if self.pts_prev is None:
      return None
    return [cv2.KeyPoint(x=p[0][0] + offset_x, y=p[0][1] + offset_y, _size=10) for p in self.pts_prev]
