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
            probability,
            label,
            predict_color=(0, 255, 0),
            true_color=(0, 0, 255)
    ):
        index = int(probability[0] < probability[1])
        cv2.putText(
            frame, f"TRUE: {'ON' if label else 'OFF'}",
            (10, IMAGE_SIZE - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, true_color, 2, cv2.LINE_8
        )
        cv2.putText(
            frame, f"PREDICT: {'ON' if index else 'OFF'} {probability[index] * 100:.2f}%",
            (10, IMAGE_SIZE - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, predict_color, 2, cv2.LINE_8
        )
