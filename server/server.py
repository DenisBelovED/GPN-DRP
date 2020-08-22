from os.path import normpath, join
from flask import Flask, render_template, Response
from core.constants import ROOT_PREFIX
from core.inference import Inference
import cv2

app = Flask(__name__)
engine = Inference(normpath(join(ROOT_PREFIX, r'rosseti/data/indicator/off_video/0.MOV')))


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    engine_gen = engine()
    while True:
        predict = next(engine_gen)
        ret, jpeg = cv2.imencode('.jpg', predict[0])
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'
        )


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, threaded=True)
