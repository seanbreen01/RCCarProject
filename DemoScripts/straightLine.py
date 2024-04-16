import time 
import smbus
import sys

# Nvidia Jetson Nano i2c Bus 0
bus = smbus.SMBus(0)
# address  setup in the Arduino script
address = 0x40

# Control theory varaible definitions
maxSpeed = int(input("Set speed 1000 - 1400 for reverse, 1600 - 2000 for forward: "))
controlTimer = 10000

corner_dict_steering = {
    "straight": [0,76,controlTimer],
    "gentleLeft": [0, 80, controlTimer],
    "gentleRight": [0, 72, controlTimer],
    "rightTrim": [0, 74, controlTimer],
    "leftTrim": [0, 78, controlTimer],
    "stop": [0, 75, controlTimer],
    "straightReverse": [0, 76, controlTimer],
    "automatedRecovery": [0, 85, 3000]
    }

corner_dict_motor = {
    "straight": [1,maxSpeed,controlTimer],
    "gentleLeft": [1, maxSpeed, controlTimer],
    "gentleRight": [1, maxSpeed, controlTimer],
    "rightTrim": [1, maxSpeed, controlTimer],
    "leftTrim": [1, maxSpeed, controlTimer],
    "stop": [1, 1500, controlTimer],
    "straightReverse": [1, 1400, controlTimer],
    "automatedRecovery": [1, 1600, 3000]
    }

# I2C communications to Arduino UNO
def writeToArduino(valueToWrite):
    bytesToWrite = []
    for value in valueToWrite:
        # Split each integer into two bytes (high byte and low byte)
        highByte = (value >> 8) & 0xFF
        lowByte = value & 0xFF
        bytesToWrite.extend([highByte, lowByte])

    # Send the byte array
    bus.write_i2c_block_data(address, 0, bytesToWrite)

def readFromArduino():
    number = bus.read_byte(address)
    # number = bus.read_byte_data(address, 1)

    #TODO update to read from bus in correct format
    return number

def main():
    print("Starting straight line")
    writeToArduino(corner_dict_steering["straight"])
    writeToArduino(corner_dict_motor["straight"])
    time.sleep(10)
    writeToArduino(corner_dict_steering["stop"])
    writeToArduino(corner_dict_motor["stop"])
    print("Straight line complete")

if __name__ == "__main__":
    main()
    sys.exit()