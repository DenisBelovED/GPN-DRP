import cv2
from core.constants import IMAGE_SIZE


class VisualManager:
    def __init__(self, win_name):
        self.win_name = win_name
        self.window = cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    def desktop_show(self, frame, delay=0):
        cv2.imshow(self.win_name, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.waitKey(delay)

    @staticmethod
    def draw_annotations(
            frame,
            pred_label,
            gt_label,
            label_map,
            predict_color=(0, 255, 0),
            true_color=(0, 0, 255)
    ):
        cv2.putText(
            frame, f"TRUE: {label_map[gt_label]}",
            (10, IMAGE_SIZE - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, true_color, 1, cv2.LINE_8
        )
        cv2.putText(
            frame, f"PREDICT: {label_map[pred_label]}",
            (10, IMAGE_SIZE - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            predict_color if pred_label == gt_label else (255, 0, 0), 1, cv2.LINE_8
        )
