import os
import sys
import cv2
import numpy as np

from sklearn.metrics import mean_squared_error

from pred import calCSpeed


def draw_mask(img):
  cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 0), -1)

  x_top_offset = 180
  x_btm_offset = 35

  poly_pts = np.array([[[640-x_top_offset, 250], [x_top_offset, 250], [x_btm_offset, 350], [640-x_btm_offset, 350]]], dtype=np.int32)
  cv2.fillPoly(img, poly_pts, (255, 255, 255))

  return img


if __name__ == "__main__":


    if len(sys.argv) != 3:
        print("invalid Number of arguments")
        exit(-1)

    cap = cv2.VideoCapture(sys.argv[1])

    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    FRAMES = frame_cnt
    if os.getenv("FRAMES") is not None:
        FRAMES = int(os.getenv("FRAMES"))

    try:
        output_file = np.loadtxt(sys.argv[2], delimiter='\n')[:FRAMES]
        scale = None
    except:
        output_file = None

    try:
      scale = float(sys.argv[2])
    except:
      scale = None

    ve = calCSpeed(FRAMES, output_file=output_file, scale=scale)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mask = np.zeros(shape=(H, W), dtype=np.uint8)
    mask.fill(255)
    draw_mask(mask)

    fs_prev = None

    try:
        while True:
            _, frame = cap.read()

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ve.proc_frame(frame_gray[130:350, 35:605], mask[130:350, 35:605])
            fs = ve.get_frames(35, 130)
            fs_prev = fs

            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= FRAMES:
                break
    except KeyboardInterrupt:
        pass

    cap.release()
    cv2.destroyAllWindows()

    preds = ve.get_preds()

    with open("test.txt", "w") as f:
        for p in preds:
            f.write(str(p) + "\n")
    mean = mean_squared_error(preds, output_file) #change this line to find mean squared error put true output value file as second argument
    # eg.... sudo python app.py test.mp4 correctoutputfile.txt
    print("Mean Squared Error", mean)
   


