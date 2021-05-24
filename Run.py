import Recog
import sys
import os


def get_result(text):
    global con
    file = open("plate.txt", "w+")
    if os.path.exists('plate.txt'):
        file.close()
        file = open("plate.txt", "w")
        file.write(text)
        file.close()
        file = open("plate.txt", "r")
        con = file.read()
        file.close()
        os.remove('plate.txt')
    else:
        print('plate.txt not found')
    # print(con)


def manage_error():
    global con
    con = "4"
    print(con)


img = str(sys.argv[1])
if __name__ == '__main__':
    Recog
    a = str(Recog.final_string)
    get_result(a)
    print(con)
# python -W ignore Run.py LicPlateImages/t1.jpg
