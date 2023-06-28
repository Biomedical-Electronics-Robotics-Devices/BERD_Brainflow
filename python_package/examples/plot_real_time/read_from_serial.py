import serial

ser = serial.Serial(
    port='COM4',
    baudrate=115200,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)


def hexlify(data):
    return ' '.join(f'{c:0>2X}' for c in data)


temp = False
mylist = []

while 1:
    x = ser.read()
    byt = hexlify(x)
    if temp:
        if byt == "41":
            temp = True
            print(byt)
            mylist.append(byt)
    else:
        mylist.append(byt)
        if byt == "C0":
            if len(mylist) > 30:
                print(mylist)
                mylist = []
                temp = False
