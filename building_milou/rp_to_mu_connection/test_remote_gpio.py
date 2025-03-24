# from gpiozero.pins.pigpio import PiGPIOFactory
# from gpiozero import RGBLED
# from time import sleep
#
# factory = PiGPIOFactory(host='192.168.4.1')
#
# led = RGBLED(red=9, green=10, blue=11, pin_factory=factory)

import pigpio
import time

import mouse_click_dantec

pi = pigpio.pi('192.168.4.1')
print("Connected to pi? ", pi.connected)

arduino = pi.serial_open("/dev/ttyACM0", 9600)
print("Connected to arduino on handle:", arduino)

try:
    while True:
        if pi.serial_data_available(arduino):
            print("data available! woop woop!")
            # do something with data
            nbytes, data = pi.serial_read(arduino, 100)
            if nbytes > 0:
                print(data)
            # convert byte data into a string
            readable_data = data.decode().strip()
            if readable_data == "18":
                print("Bingpot!")

                # run code that moves the mouse
                mouse_click_dantec.comfortSense()

                # write data to arduino
                pi.serial_write(arduino, "42")
        time.sleep(1)

finally:
    # close serial interface with arduino
    pi.serial_close(arduino)
    # release pigpio resources
    pi.stop()
