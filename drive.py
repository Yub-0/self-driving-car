import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image

# real time communication between client and server
sio = socketio.Server()

app = Flask(__name__)
speed_limit = 25


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    # forcing speed limit
    throttle = 1.0 - speed / speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)


# connect, disconnect, message event
# as soon as connection is established we send the control command for steering angle and throttle
@sio.on('connect')
def connect(sid, my_env):
    print('Connected')
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    # load the model
    model = load_model('model.h5')
    # combine sio and flask
    # middleware to dispatch the data between sio and app
    app = socketio.Middleware(sio, app)
    # create a socket
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
