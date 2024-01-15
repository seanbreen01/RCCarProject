import smbus
import time
# Nvidia Jetson Nano i2c Bus 0
bus = smbus.SMBus(0)

# This is the address we setup in the Arduino script
address = 0x40

def writeToArduino(valueToWrite):
    bytesToWrite = []
    for value in valueToWrite:
        # Split each integer into two bytes (high byte and low byte)
        highByte = (value >> 8) & 0xFF
        lowByte = value & 0xFF
        bytesToWrite.extend([highByte, lowByte])

    # Send the byte array
    print('Bytes: ')    #For debug
    print(bytesToWrite)
    bus.write_i2c_block_data(address, 0, bytesToWrite)

def readNumber():
    number = bus.read_byte(address)
    # number = bus.read_byte_data(address, 1)

    #TODO update to read from bus in correct format
    return number


if __name__ == "__main__":
    # create rough testing script

    #jiggle wheels
    jiggleWheelsValues = []
    # Format is:
    # 0 = steering servo, 1 = motor
    # 45 = angle of steering servo, or motor power in motor instance
    # 1000 = duration (in millis) to hold this position
    jiggleWheelsValues.append([0, 45, 500])  
    jiggleWheelsValues.append([0, 0, 750])
    jiggleWheelsValues.append([0, 130, 250])
    jiggleWheelsValues.append([0, 90, 800])
    jiggleWheelsValues.append([0, 165, 600])

    for value in jiggleWheelsValues:
        print('here')
        writeToArduino(value)
        
        # print(value)    #For debugging

        time.sleep(1)    #this will need to be replaced by something else as the time.sleep will stop the camera from functioning in full scale model
   
   
    #forward and back?
    #TODO need to figure out if motor can be made to run in reverse

    forwardBackward = []

    forwardBackward.append([1, 1500, 1000])
    forwardBackward.append([1, 1500, 1000])
    forwardBackward.append([1, 1500, 1000])
    forwardBackward.append([1, 1500, 1000])

    #Drive in circle

    # Another one?