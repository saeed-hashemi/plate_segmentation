import numpy as np

def calcPointsWH(pts, theta=0):
    # Measuring width of points at given angle
    th = theta * np.pi /180
    e = np.array([[np.cos(th), np.sin(th)]]).T
    es = np.array([
    [np.cos(th), np.sin(th)],
    [np.sin(th), np.cos(th)],
    ]).T
    dists = np.dot(pts,es)
    wh = dists.max(axis=0) - dists.min(axis=0)
    print("==> theta: {}\n{}".format(theta, wh))
    return True if wh[0]>wh[1] else False


def sort_rect_points(box):
    m_rect = np.zeros((4, 2), dtype="float32")
    s = box.sum(axis=1)
    s_min=np.argwhere(s == np.amin(s))
    if len(s_min)==1:
      m_rect[0] = box[np.argmin(s)]
    else:
      m_rect[0] = box[s_min[0]] if box[s_min[0]][0][0]<box[s_min[1]][0][0] else box[s_min[1]]

    s_max=np.argwhere(s == np.amax(s))
    if len(s_max)==1:
      m_rect[2] = box[np.argmax(s)]
    else:
      m_rect[2] = box[s_max[0]] if box[s_max[0]][0][0]>box[s_max[1]][0][0] else box[s_max[1]]
    
    diff = np.squeeze(np.diff(box, axis=1))
    diff_min=np.argwhere(diff == np.amin(diff))
    if len(diff_min)==1:
      m_rect[1] = box[np.argmin(diff)]
    else:
      m_rect[1] = box[diff_min[0]] if box[diff_min[0]][0][0]<box[diff_min[1]][0][0] else box[diff_min[1]]
    
    diff_max=np.argwhere(diff == np.amax(diff))
    if len(diff_max)==1:
      m_rect[3] = box[np.argmax(diff)]
    else:
      m_rect[3] = box[diff_max[0]] if box[diff_max[0]][0][0]>box[diff_max[1]][0][0] else box[diff_max[1]]
    return m_rect


def is_plate_on_left(second_point,image_width):
  print(second_point)
  return True if second_point[0]<image_width/2 else False