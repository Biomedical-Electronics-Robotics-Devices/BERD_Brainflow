import serial
from threading import Thread
import pyedflib
import numpy as np
from pyedflib import highlevel
import datetime


class Geenie:
    def __init__(self):
        self.id = None
        self.header = None
        self.edf_filename = None
        self.signals = None
        self.signal_headers = None
        self.data = None

    def new_measurement(self, patientname, recording_minutes=1, technician="", recording_additional="",
                        patient_additional="", patientcode="", equipment="Geenie", sex="",
                        startdate="", birthdate="", sampling_rate=250, number_of_channels=8,
                        channel_names=None,
                        edf_filename=datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S.edf")):

        if channel_names is None:
            channel_names = []
            for n in range(1, number_of_channels + 1):
                channel_names.append(f"ch{n}")

        # signals = np.random.rand(number_of_channels,
        #                          sampling_rate * 60 * recording_minutes) * 200  # 5 minutes of random signal
        self.data = np.zeros([number_of_channels, sampling_rate * 60 * recording_minutes])

        self.signal_headers = highlevel.make_signal_headers(channel_names, sample_frequency=sampling_rate)
        self.header = highlevel.make_header(technician=technician,
                                            recording_additional=recording_additional,
                                            patientname=patientname,
                                            patient_additional=patient_additional,
                                            patientcode=patientcode,
                                            equipment=equipment,
                                            sex=sex,
                                            startdate=startdate,
                                            birthdate=birthdate
                                            )

    def start_measurement(self):
        # TODO Fill the data table
        pass

    def save_file(self):
        highlevel.write_edf(edf_file=self.edf_filename,
                            signals=self.signals,
                            signal_headers=self.signal_headers,
                            header=self.header)


# # Please adjust all parameters before each test
# # _______________________________________________
# # Measurement Info
# recording_minutes = 5
# technician = "Vasilis Vasilopoulos"
# recording_additional = "Add comments here"
# # Patient Info
# patientcode = "1"
# patientname = "Vasilis Vasilopoulos"
# patient_additional = "Comments about the patient"
# birthdate = datetime.date(day=23, month=10, year=1992).strftime("%d-%m-%Y")
# startdate = datetime.datetime.now()
# sex = "male"
# # Equipment Info
# equipment = "Geenie"
# number_of_channels = 8
# channel_names = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']
# sampling_rate = 250
# # _______________________________________________
#
#
# # Specify the filename
# edf_filename = datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S.edf")


# ser = serial.Serial(
#     port='/dev/ttyUSB0',
#     baudrate=115200,
#     parity=serial.PARITY_NONE,
#     stopbits=serial.STOPBITS_ONE,
#     bytesize=serial.EIGHTBITS,
#     timeout=1
# )
#
#
# def hexlify(data):
#     return ' '.join(f'{c:0>2X}' for c in data)
#
#
# def read():
#     temp = False
#     mylist = []
#
#     while 1:
#         x = ser.read()
#         byt = hexlify(x)
#         if temp:
#             if byt == "41":
#                 temp = True
#                 # print(byt)
#                 mylist.append(byt)
#         else:
#             mylist.append(byt)
#             if byt == "C0":
#                 if len(mylist) > 30:
#                     print(mylist)
#                     mylist = []
#                     temp = False
#
#
if __name__ == "__main__":
    geenie = Geenie()
    geenie.new_measurement(patientname="Vasilis Vasilopoulos",
                           recording_minutes=5,
                           )
    # thread = Thread(target=read)
    # thread.start()
    # while True:
    #     comm = input()
    #     ser.write(comm.encode())
