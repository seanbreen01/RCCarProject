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
    # Format is:
    # 0 = steering servo, 1 = motor
    # 45 = angle of steering servo, or motor power in motor instance
    # 1000 = duration (in millis) to hold this position

    #jiggle wheels
    jiggleWheelsValues = []
    jiggleWheelsValues.append([0, 60, 200])  #minimum steering angle
    jiggleWheelsValues.append([0, 65, 200])
    jiggleWheelsValues.append([0, 70, 200])
    jiggleWheelsValues.append([0, 75, 200])
    jiggleWheelsValues.append([0, 80, 200])
    jiggleWheelsValues.append([0, 85, 200])  
    jiggleWheelsValues.append([0, 90, 200])
    jiggleWheelsValues.append([0, 95, 200])
    jiggleWheelsValues.append([0, 100, 200])
    jiggleWheelsValues.append([0, 105, 200])
    jiggleWheelsValues.append([0, 110, 200]) # maximum steering angle
    jiggleWheelsValues.append([0, 105, 200])
    jiggleWheelsValues.append([0, 100, 200])
    jiggleWheelsValues.append([0, 95, 200])
    jiggleWheelsValues.append([0, 90, 200])
    jiggleWheelsValues.append([0, 85, 200])
    jiggleWheelsValues.append([0, 80, 200])
    jiggleWheelsValues.append([0, 75, 200])  
    jiggleWheelsValues.append([0, 70, 200])
    jiggleWheelsValues.append([0, 65, 200])
    jiggleWheelsValues.append([0, 60, 200])


    for value in jiggleWheelsValues:
        writeToArduino(value)
        time.sleep(0.1)    #this will need to be replaced by something else as the time.sleep will stop the camera from functioning in full scale model
   
   
    #forward and back?
    #TODO need to figure out if motor can be made to run in reverse
    forwardBackward = []

    forwardBackward.append([1, 1100, 1000])
    forwardBackward.append([1, 1500, 1000])
    forwardBackward.append([1, 1500, 1000])
    forwardBackward.append([1, 1500, 1000])

    #Drive in circle
    #TODO check if addressing of motor and steering is correct, i.e. is steering sufficient and motor not running to fast
    # in essence just a real world test to make sure everything is happy
    circleDrive = []
    circleDrive.append([0, 110, 1000])
    circleDrive.append([1, 1200, 1000])

    for value in circleDrive:
        writeToArduino(value)
        time.sleep(1) # need to adjust this value appropriately 
    # Another one?
        
    