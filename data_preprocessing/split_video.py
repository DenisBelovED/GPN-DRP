from data_preprocessing.frame_manager import FrameManager
from cv2 import imwrite
from core.constants import ROOT_PREFIX
from os.path import join, normpath
from os import listdir


def write(video_path, img_path):
    for v_name in listdir(video_path):
        v_path = join(video_path, v_name)
        frame_generator = FrameManager.init_video_frame_generator(v_path)  # покадрово выдаёт видеопоток
        index = 0
        while True:
            # for t in range(39):
            #     next(frame_generator)
            try:
                imwrite(normpath(join(img_path, f"{index}.jpg")), next(frame_generator))
            except StopIteration:
                break
            index += 1


def main():
    off_video_path = normpath(join(ROOT_PREFIX, r'rosseti/data/indicator/off_video'))
    on_video_path = normpath(join(ROOT_PREFIX, r'rosseti/data/indicator/on_video'))
    off_img_path = normpath(join(ROOT_PREFIX, r'rosseti/data/indicator/off'))
    on_img_path = normpath(join(ROOT_PREFIX, r'rosseti/data/indicator/on'))
    write(off_video_path, off_img_path)
    write(on_video_path, on_img_path)


if __name__ == '__main__':
    main()
