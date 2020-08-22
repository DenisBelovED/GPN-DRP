from help_scripts.frame_manager import FrameManager
from dev.visual_manager import VisualManager
from hashlib import md5
from cv2 import imwrite, waitKey
from dev.constants import ROOT_PREFIX


def main():
    video_name = ROOT_PREFIX + r'cv_gpn_transport_detection/data/videos/unuse/.avi'
    frame_generator = FrameManager.init_video_frame_generator(video_name)  # покадрово выдаёт видеопоток

    vm = VisualManager("frame")
    while True:
        frame = next(frame_generator)
        vm.desktop_show(frame)
        code = waitKey(0)
        if code == 115:  # нажата ли клавиша s, то сохранить кадр
            imwrite(
                ROOT_PREFIX + r'cv_gpn_transport_detection/data/unused_images/dataset_img/' +
                str(md5(frame).hexdigest()) +
                r'.jpg',
                frame
            )
        if (48 < code) and (code < 58):  # если нажата клавиша 1-9, то пропустить столько-же кадров
            for i in range(code - 48):
                next(frame_generator)
        if code == 113:  # если нажали q, то брякаемся из цикла
            break


if __name__ == '__main__':
    main()
