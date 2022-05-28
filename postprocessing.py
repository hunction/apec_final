import numpy as np
import sympy as sp
import base64
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def targetchanger(y_hat):
    index = ["/", "=", "!", "integral", "[", ">=", ">", "{", "(", "-", "*", "+", "]", "<=", "<", "}", ")"]
    target = []
    # target값을 정수형으로 반환
    for i in range(len(y_hat)):
        target.append(np.argmax(y_hat[i]))

    for j in range(len(target)):
        if target[j] < 10:
            target[j] = "%d" % target[j]

        elif (9 < target[j] and target[j] < 62):

            target[j] = "%c" % (55 + target[j])

        elif target[j] > 61:

            target[j] = index[target[j] - 62]

        else:

            print("out of bound")

        # 0~9 : 숫자

        # 10~35 : 대문자 Alphabet A~Z

        # 36~61 : 소문자 Alphabet a~z

        # 62~78 : 기호  /, =, ! , integral, [ ,>=, > , { , ( , - , * , + , ] , <= , < , } , )

        # 총 79개의 Index

    return target


def mnisttargetchanger(y_hat):
    index = ["/", "=", "!", "integral", "[", ">=", ">", "{", "(", "-", "*", "+", "]", "<=", "<", "}", ")"]
    target = []
    # target값을 정수형으로 반환
    for i in range(len(y_hat)):
        target.append(np.argmax(y_hat[i]))

    for j in range(len(target)):
        target[j] = "%d" % target[j]

    return target

def Bigtargetchanger(y_hat):
    index = ["/", "=", "!", "integral", "[", ">=", ">", "{", "(", "-", "*", "+", "]", "<=", "<", "}", ")"]
    target = []
    # target값을 정수형으로 반환
    for i in range(len(y_hat)):
        target.append(np.argmax(y_hat[i]))

    for j in range(len(target)):
        target[j] = "%c" % (65 + target[j])

    return target


def smalltargetchanger(y_hat):
    index = ["/", "=", "!", "integral", "[", ">=", ">", "{", "(", "-", "*", "+", "]", "<=", "<", "}", ")"]
    target = []
    # target값을 정수형으로 반환
    for i in range(len(y_hat)):
        target.append(np.argmax(y_hat[i]))

    for j in range(len(target)):
        target[j] = "%c" % (97 + target[j])

    return target

def indextargetchanger(y_hat):
    index = ["/", "=", "!", "integral", "[", ">=", ">", "{", "(", "-", "*", "+", "]", "<=", "<", "}", ")"]
    target = []
    # target값을 정수형으로 반환
    for i in range(len(y_hat)):
        target.append(np.argmax(y_hat[i]))

    for j in range(len(target)):
        target[j] = index[target[j]]

    return target


def solve(target, x_cls):
    index = ["/", "=", "!", "integral", "[", ">=", ">", "{", "(", "-", "*", "+", "]", "<=", "<", "}", ")"]

    function = ['cos', 'sin', 'tan', 'f(x)', 'log', 'ln', 'C', 'P', 'e', 'cosh', 'sinh', 'tanh', 'integral', 'i']

    spfunction = ['sp.cos', 'sp.sin', 'sp.tan', 'f(x)', 'log', 'ln', 'C', 'P', 'sp.exp', 'sp.cosh', 'sp.sinh',
                  'sp.tanh', 'integrate', 'sp.I']

    function2 = ['cos', 'sin', 'tan', 'cosh', 'sinh', 'tanh', 'log', 'ln']

    num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    small_letter = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
                    "u", "v", "w", "x", "y", "z"]

    large_letter = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                    "U", "V", "W", "X", "Y", "Z"]

    letter = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
              "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
              "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

    bracket = ["[", "{", "(", ")", "}", "]"]
    # sympy 변수지정 : C P i e 네개 제외
    a, b, c, d, f, g, h, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, A, B, D, E, F, G, H, I, J, K, L, M, N, O, Q, R, S, T, U, V, W, X, Y, Z = sp.symbols(
        "a b c d f g h j k l m n o p q r s t u v w x y z A B D E F G H I J K L M N O Q R S T U V W X Y Z")

    target1 = ''.join(target)
    string = []
    x_cls1 = []
    count = len(target1)

    for j in range(1, len(x_cls)-1):
        # len(x_cls)=len(초기 target)
        if x_cls[j - 1] == 0 and x_cls[j] == 2 and x_cls[j + 1] == 0:
            x_cls1 = np.append(x_cls[:j], ['0'])
            x_cls = np.append(x_cls1, x_cls[j:])
            string1 = np.append(target1[:j], ['**'])
            target1 = np.append(string1, target1[j:])
            j += 1

        elif x_cls[j - 1] == 0 and x_cls[j] == 2 and x_cls[j + 1] == 2:
            x_cls1 = np.append(x_cls[:j], ['0', '0'])
            x_cls = np.append(x_cls1, x_cls[j:])
            string1 = np.append(target1[:j], ['**', '('])
            target1 = np.append(string1, target1[j:])
            j += 2

        elif x_cls[j - 1] == 2 and x_cls[j] == 2 and x_cls[j + 1] == 0:
            x_cls1 = np.append(x_cls[:j + 1], ['0'])
            x_cls = np.append(x_cls1, x_cls[j + 1:])
            string1 = np.append(target1[:j + 1], [')'])
            target1 = np.append(string1, target1[j + 1:])

        elif x_cls[j-1] == 2 and x_cls[j] == 2 and j == len(x_cls):
            x_cls1 = np.append(x_cls[:j + 1], ['0'])
            x_cls = np.append(x_cls1, x_cls[j + 1:])
            string1 = np.append(target1[:j + 1], [')'])
            target1 = np.append(string1, target1[j + 1:])


    target1 = ''.join(target1)
    # function algorithm
    while count > 0:
        for i in range(len(target1)):
            if target1[:-i] in function:
                string = np.append(string, target1[:-i])
                target1 = target1[-i:]
                count = count - (len(target1) - i)
                i = 0

            elif (len(target1[:-i]) == 1):
                string = np.append(string, target1[:-i])
                target1 = target1[-i:]
                count = count - 1
                i = 0

            elif (len(target1) == 1):
                string = np.append(string, target1)
                target1 = []
                count = 0


    string1 = []
    string2 = []
    string3 = []
    string4 = []
    string5 = []
    string6 = []
    string7 = []
    string8 = []
    # integral to integrate
    for j in range(len(string)):
        if string[j] == "integral" and string[j + 1] == "(":
            string[j] = "integrate"
            for i in range(len(string)):
                if string[i] == ")" and string[i + 1] == "d":
                    string1 = np.append(string[:j + 1], "(")
                    string[i + 1] = ","
                    string2 = np.append(string[j + 1:i + 3], ")")
                    string3 = np.append(string1, string2)
                    string = np.append(string3, string[i + 3:])
                    break
        elif string[j] == "i":
            string[i] = "I"
        elif string[j] == "log":
            for h in range(j, len(string)):
                if string[h] == "(":
                    for k in range(h, len(string)):
                        if string[k] == ")":
                            string4 = np.append(string[:j + 1], string[h:k])
                            string5 = np.append(",", string[j + 1:h])
                            string6 = np.append(string4, string5)
                            string = np.append(string6, string[k:])
                            break
                    break

    string1 = []

    for i in range(len(string) - 1):
        # 알파벳 뒤에 숫자
        for j in range(len(letter)):
            for k in range(len(num)):
                if (letter[j] == string[i]) and (num[k] == string[i + 1]):
                    string1 = np.append(string[:i + 1], ['*'])
                    string = np.append(string1, string[i + 1:])

    for i in range(1, len(string)):
        # 알파벳 앞에 숫자
        for j in range(len(letter)):
            for k in range(len(num)):
                if (letter[j] == string[i]) and (num[k] == string[i - 1]):
                    string1 = np.append(string[:i], ['*'])
                    string = np.append(string1, string[i:])

    string1 = []
    string2 = []
    for i in range(len(string) - 1):
        for j in range(len(letter)):
            if (letter[j] == string[i]) and (string[i + 1] == bracket[0]):
                string1 = np.append(string[:i + 1], ['*'])
                string = np.append(string1, string[i + 1:])

            elif (letter[j] == string[i]) and (string[i + 1] == bracket[1]):
                string1 = np.append(string[:i + 1], ['*'])
                string = np.append(string1, string[i + 1:])

            elif (letter[j] == string[i]) and (string[i + 1] == bracket[2]):
                string1 = np.append(string[:i + 1], ['*'])
                string = np.append(string1, string[i + 1:])

        for k in range(len(num)):
            if (num[k] == string[i]) and (string[i + 1] == bracket[0]):
                string2 = np.append(string[:i + 1], ['*'])
                string = np.append(string2, string[i + 1:])

            elif (num[k] == string[i]) and (string[i + 1] == bracket[1]):
                string2 = np.append(string[:i + 1], ['*'])
                string = np.append(string2,string[i + 1:])

            elif (num[k] == string[i]) and (string[i + 1] == bracket[2]):
                string2 = np.append(string[:i + 1], ['*'])
                string = np.append(string2,string[i + 1:])

    string1 = []

    for i in range(1, len(string)):
        for j in range(len(letter)):
            if (letter[j] == string[i]) and (string[i - 1] == bracket[3]):
                string1 = np.append(string[:i], ['*'])
                string = np.append(string1, string[i:])

            elif (letter[j] == string[i]) and (string[i - 1] == bracket[4]):
                string1 = np.append(string[:i], ['*'])
                string = np.append(string1, string[i:])

            elif (letter[j] == string[i]) and (string[i - 1] == bracket[5]):
                string1 = np.append(string[:i], ['*'])
                string = np.append(string1, string[i:])

        for k in range(len(num)):
            if (num[k] == string[i]) and (string[i - 1] == bracket[3]):
                string1 = np.append(string[:i], ['*'])
                string = np.append(string1, string[i:])

            elif (num[k] == string[i]) and (string[i - 1] == bracket[4]):
                string1 = np.append(string[:i], ['*'])
                string = np.append(string1, string[i:])

            elif (num[k] == string[i]) and (string[i - 1] == bracket[5]):
                string1 = np.append(string[:i], ['*'])
                string = np.append(string1, string[i:])

    # 문자 문자 곱셈
    for i in range(len(string) - 1):
        for j in range(len(letter)):
            for k in range(len(letter)):
                if (string[i] == letter[j]) and (string[i + 1] == letter[k]):
                    string1 = []
                    string1 = np.append(string[:i + 1], index[10])
                    string = np.append(string1, string[i + 1:])

    # sympy 사용전 수식화
    arr = ''.join(string)
    arr2 = []
    arr2 = np.append(arr2, arr)
    arr3 = []
    cnt = 0
    errormsg = "계산이 불가능한 수식 입니다."

    try:
        expr = sp.sympify(arr2[0])


        return str(expr) , string

    except:
        return errormsg , string


def Graph(expr):

    arr = np.zeros((200,500,4))
    try:
        graph = sp.plot(expr,show=False)
        graph.save("./graph_img/graphtest.png")
        path = "./graph_img/graphtest.png"
        graph_rs = cv2.imread(path)
        graph_rs = cv2.resize(graph_rs, (160, 120))
        b = graph_rs[:, :, 0]
        g = graph_rs[:, :, 1]
        r = graph_rs[:, :, 2]
        result = ((0.299 * r) + (0.587 * g) + (0.114 * b))
        # imshow 는 CV_8UC3 이나 CV_8UC1 형식을 위한 함수이므로 타입변환
        result = result.astype(np.uint8)
        cv2.imwrite("./graph_img/graph1.png" , result)
        path_1 = "./graph_img/graph1.png"
        with open(path_1, 'rb') as f:
            data = f.read()
        return data

    except:
        arr.save("./graph_img/error.png")

        path = "./graph_img/error.png"
        with open(path, 'rb') as f:
            data = f.read()

        return data

def variable(target, function_data):

    var = []
    res = []

    for i in range(len(target)):
        for j in range(len(target)):
            if (target[i]=="=" and target[j]=="=" and i < j):
                var = np.append(var, target[i-1])
                res = np.append(res, target[i+1:j-1])
            elif (target[i]=="=" and i<j and j== len(target)):
                var=np.append(var,target[i-1])
                res=np.append(res,target[i+1:])

    if len(var) == 1:
        expr = sp.sympify(function_data)
        exprk = expr.subs(var[0], res[0])
    elif len(var) == 2:
        expr = sp.sympify(function_data)
        exprk = expr.subs([(var[0], res[0]),(var[1], res[1])])
    elif len(var) == 3:
        expr = sp.sympify(function_data)
        exprk = expr.subs([(var[0], res[0]), (var[1], res[1]),(var[2], res[2])])
    elif len(var) == 3:
        expr = sp.sympify(function_data)
        exprk = expr.subs([(var[0], res[0]), (var[1], res[1]), (var[2], res[2]), (var[3], res[3])])
    elif len(var) == 3:
        expr = sp.sympify(function_data)
        exprk = expr.subs([(var[0], res[0]), (var[1], res[1]), (var[2], res[2]), (var[3], res[3]), (var[4], res[4])])

    return exprk, function_data







