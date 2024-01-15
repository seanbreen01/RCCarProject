import smbus
import time
# Nvidia Jetson Nano i2c Bus 0
bus = smbus.SMBus(0)

# This is the address we setup in the Arduino script
address = 0x40

def writeToArduino(valueToWrite):
    
    # bus.write_byte(address, int(value))
    

    #TODO update this so that values are sent in correct format for Arduino to receive
    # value will be a list of numbers that inform the vehicle of how to react


#Will need to check the format of this no doubt
    bus.write_i2c_block_data(address, 0, [valueToWrite[0], valueToWrite[1], valueToWrite[2]])

    return -1

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
        writeToArduino(jiggleWheelsValues[value])
        
        print(jiggleWheelsValues[value])    #For debugging

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