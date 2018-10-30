import math

def vFromM(mass, temp, tempUnit = "C"):
    try:
        mass = float(mass)
        temp = float(temp)
    except ValueError:
        print("Input mass and/or temperature can not be converted to floats.")
    volume = (mass / density(temp, tempUnit))
    return volume

def density(temp, tempUnit = "C"):
    try:
        temp = float(temp)
    except ValueError:
        print("Input temperature can not be converted to floats.")
    if tempUnit == "C":
        temp = temp + 273.15
    elif tempUnit == "K":
        temp = temp
    else:
        pass
    density = 0.14395 / (math.pow(0.0112, (1+ math.pow(1-(temp/649.727), 0.05107))))
    return density
