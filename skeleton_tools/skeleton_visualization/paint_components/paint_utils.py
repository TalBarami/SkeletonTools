import cv2
import numpy as np
from matplotlib import pyplot as plt

from skeleton_tools.utils.constants import EPSILON


# def draw_bbox(frame, bbox, bcolor=(255, 255, 255)):
#     center, r = bbox
#     r = r // 2
#     if frame.shape[-3] > 3:
#         bcolor += (255,)
#     cv2.rectangle(frame, tuple((center - r).astype(int)), tuple((center + r).astype(int)), color=bcolor, thickness=1)
#
#
# def draw_pid(frame, pose, c, pid, color):
#     if np.all(c < EPSILON):
#         return
#     x = pose[0][c > EPSILON]
#     y = pose[1][c > EPSILON]
#     x_center = x.mean() * 0.975
#     y_center = y.min() * 0.9
#     cv2.putText(frame, str(pid), (int(x_center), int(y_center)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)
#
#
def blur_area(frame, center, radius):
    c_mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.circle(c_mask, center, radius, 1, thickness=-1)
    mask = cv2.bitwise_and(frame, frame, mask=c_mask)
    img_mask = frame - mask
    blur = cv2.blur(frame, (50, 50))
    mask2 = cv2.bitwise_and(blur, blur, mask=c_mask)  # mask
    final_img = img_mask + mask2
    return final_img

def fig2np(fig):
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
