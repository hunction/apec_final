import cv2
import numpy as np





def padding(img):
    h, w ,c = img.shape

    set_size = max(h, w)

    if (h > w):
        delta_w = set_size - w
        delta_h = h - set_size
    elif (h < w):
        delta_w = w - set_size
        delta_h = set_size - h
    elif (h == w):
        return img

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)

    left, right = delta_w // 2, delta_w - (delta_w // 2)

    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return new_img


def slicing(X):
    Y = []

    for l in range(len(X)):
        new = np.zeros((X[l].shape[0] + 10, X[l].shape[1]))
        new[5:-5, :] = X[l]

        e = 0
        f = 1

        for f in range(len(new) - 1):
            if (new[f, :] == 0).all() and (new[f + 1, :] > 0).any():
                start = f
                for e in range(len(new) - 1):
                    if (new[-(e + 1), :] == 0).all() and (new[-(e + 2), :] > 0).any():
                        end = len(new) - e
                        break
                break

        org1 = new[start - 1:end + 1, ]
        Y.append(org1)

    return Y


def cutting(imgArray):
    X = []
    X_1 = []
    X_2 = []
    color_label = []
    x_cls = []


    s = imgArray[:,:,3]

    new_s = np.zeros((s.shape[0], s.shape[1] + 10))
    new_s[:, :-10] = s

    i = 0
    j = 0

    for j in range(len(new_s[0, :]) - 1):
        if (new_s[:, j] == 0).all() and (new_s[:, j + 1] > 0).any():
            for i in range(j + 1, len(new_s[0, :]) - 1):
                if (new_s[:, i] > 0).any() and (new_s[:, i + 1] == 0).all():
                    X_1.append((j,i))
                    img1 = new_s[:, j - 1:i + 1]
                    X.append(img1)
                    break

    for i in range(len(X)):
        hf = int(len(X[i]) / 2)
        idx = X[i]
        # 아랫첨자
        if ((idx[:hf, :] > 0).any() and (idx[hf:-1,:] == 0 ).all() ):
            x_cls.append(2)
        elif ((idx[hf:, :]> 0).any()  and (idx[0:hf, :] == 0).all()):
            x_cls.append(1)
        else:
            x_cls.append(0)

    for j,i in X_1:
        X_2.append(imgArray[:,j:i,:])

    for i in range(len(X_2)):
        if (X_2[i][:, :, 0].all() == 0 and X_2[i][:, :, 1].all() == 0 and X_2[i][:, :, 2].all() == 0):
            color_label.append(1) # 검정
        else:
            color_label.append(0)

    for i in range(len(X_2)):
        if (X_2[i][:, :, 0].any() != 0 and X_2[i][:, :, 1].all() == 0 and X_2[i][:, :, 2].all() == 0):
            color_label[i] = 2  # 파랑

    for i in range(len(X_2)):
        if (X_2[i][:, :, 0].all() == 0 and X_2[i][:, :, 1].any() != 0 and X_2[i][:, :, 2].all() == 0):
            color_label[i] = 3  # 초록

    for i in range(len(X_2)):
        if (X_2[i][:, :, 0].all() == 0 and X_2[i][:, :, 1].all() == 0 and X_2[i][:, :, 2].any() != 0):
            color_label[i] = 4  # 빨강



    return X , color_label , x_cls


def change(Y):
    Z =[]

    for prd in Y:

        data = np.expand_dims(prd , axis=-1)

        data = padding(data)
        data = cv2.resize(data, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)

        Z.append(data)  # 각각의 image data를 갖는 X배열 생성

    Z = [cv2.resize(image, (32, 32)) for image in Z]  # model에 넣기 위한 data 변환과정
    Z = np.array(Z, dtype="float32")
    Z = np.expand_dims(Z, axis=-1)
    Z /= 255.0

    return Z