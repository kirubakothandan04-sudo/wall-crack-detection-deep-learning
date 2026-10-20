"""
annotate_cracks.py
Draw the crack on a photo and save <photo>_mask.png (white = crack).
Usage: python annotate_cracks.py photo1.jpg photo2.jpg ...
Click the window first. Left-click = add a point along the crack.
n = finish this stroke and start a new one, u = undo last stroke,
s = save the mask and go to the next photo, q = quit.
"""
import os
import sys
import cv2
import numpy as np

for p in sys.argv[1:]:
    img = cv2.imread(p)
    h, w = img.shape[:2]
    sc = min(1.0, 900 / max(h, w))
    img = cv2.resize(img, (int(w * sc), int(h * sc)))
    strokes, cur = [], []

    def draw():
        vis = img.copy()
        for s in strokes + ([cur] if cur else []):
            if len(s) > 1:
                cv2.polylines(vis, [np.array(s)], False, (0, 255, 0), 2)
            for pt in s:
                cv2.circle(vis, pt, 3, (0, 0, 255), -1)
        cv2.imshow("annotate", vis)

    def mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            cur.append((x, y))
            draw()

    cv2.namedWindow("annotate")
    cv2.setMouseCallback("annotate", mouse)
    draw()
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == ord("n") and cur:
            strokes.append(cur[:])
            cur.clear()
            draw()
        elif k == ord("u") and strokes:
            strokes.pop()
            draw()
        elif k == ord("s"):
            if cur:
                strokes.append(cur[:])
                cur.clear()
            m = np.zeros(img.shape[:2], np.uint8)
            for s in strokes:
                if len(s) > 1:
                    cv2.polylines(m, [np.array(s)], False, 255, 7)
            cv2.imwrite(os.path.splitext(p)[0] + "_mask.png", m)
            print("saved", os.path.splitext(p)[0] + "_mask.png")
            break
        elif k == ord("q"):
            cv2.destroyAllWindows()
            sys.exit()
cv2.destroyAllWindows()