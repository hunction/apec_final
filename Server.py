import socket

import cv2.cv2
from PIL import Image
from PIL import ImageFile
import io
import base64
import os
import matplotlib.pyplot as plt
import glob
import numpy as np
import sympy as sp
import cv2
from keras.models import load_model
import preprocessing as prp
import postprocessing as ptp


model_index = load_model('ResNet_index.h5')
model_small = load_model('ResNet_small.h5')
model_mnist = load_model('Resnet_mnist.h5')
model_Big   = load_model('Resnet_Big.h5')

ImageFile.LOAD_TRUNCATED_IMAGES = True

host = '192.168.0.100'
#host = '172.17.70.139'
port = 9999

server_sock = socket.socket(socket.AF_INET)
server_sock.bind((host, port))
server_sock.listen(1)


while True:
    print("기다리는 중")
    client_sock, addr = server_sock.accept()

    print('Connected by', addr)
    data = client_sock.recv(1024)
    data = data.decode()
    print(data)

    while True:
        # 클라이언트에서 받을 문자열의 길이
        data = client_sock.recv(4)
        length = int.from_bytes(data, "big")
        imgdata = client_sock.recv(length)
        # data를 더 이상 받을 수 없을 때
        if ( len(imgdata) < length ):
            imgdata += client_sock.recv(length)

        '''msg64 = msg.decode()
        msg64 = msg64 + '=' * (4-len(msg64) % 4)
        imgdata = base64.b64decode(msg64)'''
        dataBytesIO = io.BytesIO(imgdata)
        image = Image.open(dataBytesIO)
        imgArray = np.array(image)


        '''function_data = client_sock.recv(1024)
        function_data = function_data.decode()'''


        X = []
        Y = []
        Z = []
        target = []
        color_label = []
        x_cls = []

        X , color_label , x_cls = prp.cutting(imgArray)
        Y = prp.slicing(X)
        Z = prp.change(Y)

        y_hat = []
        y_hat_1 = []
        y_hat_2 = []
        y_hat_3 = []
        y_hat_4 = []

        y_hat_1 = model_mnist.predict(Z)  # test image의 target값을 predict
        y_hat_2 = model_Big.predict(Z)
        y_hat_3 = model_small.predict(Z)
        y_hat_4 = model_index.predict(Z)


        target   =  ptp.mnisttargetchanger(y_hat_1)
        target_2 =  ptp.Bigtargetchanger(y_hat_2)
        target_3 =  ptp.smalltargetchanger(y_hat_3)
        target_4 =  ptp.indextargetchanger(y_hat_4)
        for i in range(len(color_label)):
            if ( color_label[i] == 1 ):
                target[i] = target[i]
            elif ( color_label[i] == 2 ):
                target[i] = target_2[i]
            elif ( color_label[i] == 3 ):
                target[i] = target_3[i]
            elif ( color_label[i] == 4 ):
                target[i] = target_4[i]

        if ("=" in target):

            function_data = client_sock.recv(1024)
            function_data = function_data.decode()

            solved, solved_t = ptp.solve(target, function_data)
            graph = ptp.Graph(solved)
            graph_str = cv2.imencode(graph)

            input_expr = solved_t

            msg_1 = " ".join(solved_t)
            msg_2 = solved
            msg_3 = graph_str

            data_1 = msg_1.encode()
            data_2 = msg_2.encode()
            data_3 = msg_3

            length_1 = len(data_1)
            length_2 = len(data_2)
            length_3 = len(data_3)

            client_sock.sendall(length_1.to_bytes(4, byteorder="little"))
            client_sock.sendall(data_1)
            client_sock.sendall(length_2.to_bytes(4, byteorder="little"))
            client_sock.sendall(data_2)
            client_sock.sendall(length_3.to_bytes(4, byteorder="little"))
            client_sock.sendall(data_3)



        else:

            solved, solved_t = ptp.solve(target, x_cls)
            graph = ptp.Graph(solved)

            input_expr = solved_t

            msg_1 = " ".join(solved_t)
            msg_2 = solved
            msg_3 = graph

            data_1 = msg_1.encode()
            data_2 = msg_2.encode()
            data_3 = msg_3

            length_1 = len(data_1)
            length_2 = len(data_2)
            length_3 = len(data_3)

            client_sock.sendall(length_1.to_bytes(4, byteorder="little"))
            client_sock.sendall(data_1)
            client_sock.sendall(length_2.to_bytes(4, byteorder="little"))
            client_sock.sendall(data_2)
            client_sock.sendall(length_3.to_bytes(4, byteorder="little"))
            client_sock.sendall(data_3)



        '''solved, solved_t = ptp.solve(target, x_cls)
        graph = ptp.Graph(solved)
        graph_str = cv2.imencode(graph)


        input_expr = solved_t



        msg_1 = " ".join(solved_t)
        msg_2 = solved
        msg_3 = graph_str

        data_1 = msg_1.encode()
        data_2 = msg_2.encode()
        data_3 = msg_3

        length_1 = len(data_1)
        length_2 = len(data_2)
        length_3 = len(data_3)

        client_sock.sendall(length_1.to_bytes(4, byteorder="little"))
        client_sock.sendall(data_1)
        client_sock.sendall(length_2.to_bytes(4, byteorder="little"))
        client_sock.sendall(data_2)
        client_sock.sendall(length_3.to_bytes(4, byteorder="little"))
        client_sock.sendall(data_3)'''



    client_sock.close()

server_sock.close()
