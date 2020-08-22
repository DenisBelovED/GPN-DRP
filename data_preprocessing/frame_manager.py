import cv2


class FrameManager:
    @staticmethod
    def init_video_frame_generator(stream_name):
        # stream_name - строка, пудь до видео

        # открываем видеопоток
        video_stream = cv2.VideoCapture(stream_name)

        # если он не открылся, выходим
        if not video_stream.isOpened():
            raise IOError(r'Видеопоток не загружен')

        # логика выдачи кадров
        while True:
            _, frame = video_stream.read()

            if _:
                yield frame
            else:
                break

        # зачищаем следы
        video_stream.release()
