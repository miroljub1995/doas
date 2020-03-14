from __future__ import division
import time

# Import the PCA9685 module.
import Adafruit_PCA9685

class Servo:
    def __init__(self):
        self.pwm = Adafruit_PCA9685.PCA9685() #0x70 - 408 motorn fungerar
        self.pwm.set_pwm_freq(60)
        self.servo_min = 307
        self.servo_max = 424

    def set(self, s):
        if s > self.servo_max:
            print("Warning! Reset servo to max ({}), was {}".format(self.servo_max, s))
            s = self.servo_max
        if s < self.servo_min:
            print("Warning! Reset servo to min ({}), was {}".format(self.servo_min, s))
            s = self.servo_min
        self.pwm.set_pwm(0, 0, s)


def test():
    s = Servo()
    curr = 366
    while True:
        print("Setting: {}".format(curr))
        s.set(curr)
        a = input("Unesite broj 8 (+ +), 2(- 1): ")
        if a == 8:
            curr += 1
        else:
            curr -= 1

#test()
