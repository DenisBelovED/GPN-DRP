from flask import Flask, render_template, Response
from core.inference import Inference
import cv2

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def gen(inference_engine):
    engine_gen = inference_engine()
    try:
        while True:
            predict = next(engine_gen)
            ret, jpeg = cv2.imencode('.jpg', predict)
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'
            )
    except StopIteration:
        print('Engine worked')


@app.route('/video_feed')
def video_feed():
    engine = Inference()
    return Response(gen(engine), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='127.0.0.1', threaded=True)
