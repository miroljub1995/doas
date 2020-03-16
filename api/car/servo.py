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
        s = int(s)
        if s > self.servo_max:
            print("Warning! Reset servo to max ({}), was {}".format(self.servo_max, s))
            s = self.servo_max
        if s < self.servo_min:
            print("Warning! Reset servo to min ({}), was {}".format(self.servo_min, s))
            s = self.servo_min
        self.pwm.set_pwm(0, 0, s)

    def set_from_range(self, val, val_range):# (-05, [-1, 1])
        res_val_nums = self.servo_max - self.servo_min
        pass_val_nums = val_range[1] - val_range[0]

        # val -= range[0]
        # val /= pass_val_nums
        # val *= res_val_nums
        # val += self.servo_min

        val = (val - val_range[0]) / pass_val_nums * res_val_nums + self.servo_min
        self.set(val)


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
