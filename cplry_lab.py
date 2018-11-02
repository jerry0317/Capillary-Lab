#!/usr/bin/env python3

# Capillary Lab Data Processing Tool
# Copyright by Jerry Yan, 2018

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math
import traceback
from statistics import mean, stdev, StatisticsError
import chi2test as cst
from prettytable import PrettyTable
from scipy.optimize import curve_fit


class CplryData(object):
    tubeLength = 0.0 # inch
    tubeDiameter = 0.0 # mm
    heightDiff = 0.0 # cm #Height Difference between the top surface and the bottom of the tube
    flowTime = 0.0 # s
    flowVolumeBefore = 0.0 # mL
    flowVolumeAfter = 0.0 # mL
    notes = ""

    def flowVolumeDiff(self):
        return (self.flowVolumeAfter - self.flowVolumeBefore)

    def tubeRadius(self):
        return (self.tubeLength)

    def flowPressure(self):
        return (997 * 9.8 * (self.heightDiff - (self.tubeLength * 2.54)) / 100)

    def flowRate(self):
        return (self.flowVolumeDiff() / self.flowTime)


class CplrySingleGroupData(CplryData):
    groupedBy = ""
    flowRates = []
    flowTimes = []
    flowVolumeDiffs = []

    def avgFlowRate(self):
        return mean(self.flowRates)

    def std(self):
        try:
            std = stdev(self.flowRates)
        except StatisticsError:
            std = 0
        return std

    def stdErr(self):
        n = len(self.flowRates)
        stdErr = self.std() / math.sqrt(n)
        return stdErr


class CplryDataSet:
    name = "Untitled"
    data = []
    uncertaintyInT = 0.005
    uncertaintyInV = 1.0

    def __init__(self, name = "Untitled"):
        self.name = name

    def defaultFilePath(self):
        fileName = self.name + ".csv"
        dirname = os.path.dirname(os.path.abspath(__file__))
        csvFilePath = os.path.join(dirname, "csv", fileName)
        return csvFilePath

    def defaultFileHeader(self):
        return ["Tube Length", "Tube Diameter", "Height Difference", "Flow Time", "Initial Volume", "Final Volume", "Notes"]

    def openFrom(self, path="DEFAULT"):
        if path == "DEFAULT":
            csvPath = self.defaultFilePath()

        csvFile = open(csvPath, "r")
        dict_reader = csv.DictReader(csvFile)

        for row in dict_reader:
            dataRow = CplryData()
            h = self.defaultFileHeader()
            dataRow.tubeLength = float(row[h[0]])
            dataRow.tubeDiameter = float(row[h[1]])
            dataRow.heightDiff = float(row[h[2]])
            dataRow.flowTime = float(row[h[3]])
            dataRow.flowVolumeBefore = float(row[h[4]])
            dataRow.flowVolumeAfter = float(row[h[5]])
            dataRow.notes = str(row[h[6]])
            self.data.append(dataRow)

        print("The data set has been successfully loaded from CSV file.")

    def save(self, option=0):
        csvFile = open(self.defaultFilePath(), "w")
        h = self.defaultFileHeader()
        dict_writer = csv.DictWriter(csvFile, h)

        dict_writer.writeheader()

        for row in self.data:
            dict_writer.writerow({
            h[0]: row.tubeLength,
            h[1]: row.tubeDiameter,
            h[2]: row.heightDiff,
            h[3]: row.flowTime,
            h[4]: row.flowVolumeBefore,
            h[5]: row.flowVolumeAfter,
            h[6]: ("N/A" if str(row.notes) == "" else str(row.notes))
            })

        csvFile.close()
        print("File has been successfully saved.")

    def add(self, option=0):
        print("You initiated a new row of data.")
        data = CplryData()
        while True:
            try:
                tLength = input("Enter the tube length in inch: ")
                data.tubeLength = float(tLength)
                break
            except ValueError:
                print("Invalid input. Please try again.")
        while True:
            try:
                tDiameter = input("Enter the tube diamter in mm: ")
                data.tubeDiameter = float(tDiameter)
                break
            except ValueError:
                print("Invalid input. Please try again.")
        while True:
            try:
                hDiff = input("Enter the height difference in cm: ")
                data.heightDiff = float(hDiff)
                break
            except ValueError:
                print("Invalid input. Please try again.")
        while True:
            try:
                fTime = input("Enter the flow time in s: ")
                data.flowTime = float(fTime)
                break
            except ValueError:
                print("Invalid input. Please try again.")
        while True:
            opt = input("Choose from the following options:\n1 - Enter flow volumes directly\n2 - Enter flow masses to get volumes\nYour choice: ")
            try:
                if opt == "1":
                    while True:
                        try:
                            volumeI = input("Enter the initial volume in mL: ")
                            data.flowVolumeBefore = float(volumeI)
                            break
                        except ValueError:
                            print("Invalid input. Please try again.")
                    while True:
                        try:
                            volumeF = input("Enter the final volume in mL: ")
                            data.flowVolumeAfter = float(volumeF)
                            break
                        except ValueError:
                            print("Invalid input. Please try again.")
                    pass
                elif opt == "2":
                    # while True:
                    #     try:
                    #         rTemp = input("Enter the room temperature in Celsius (25 by Default): ")
                    #         if rTemp == "":
                    #             roomTemp = 25.0
                    #         else:
                    #             roomTemp = float(rTemp)
                    #     except ValueError:
                    #         print("Invalid input. Please try again.")
                    while True:
                        try:
                            massI = input("Enter the initial mass in g: ")
                            flowMassBefore = float(massI)
                            data.flowVolumeBefore = (flowMassBefore / 0.997)
                            break
                        except ValueError:
                            print("Invalid input. Please try again.")
                    while True:
                        try:
                            massF = input("Enter the final mass in g: ")
                            flowMassAfter = float(massF)
                            data.flowVolumeAfter = (flowMassAfter/ 0.997)
                            break
                        except ValueError:
                            print("Invalid input. Please try again.")
                else:
                    raise Exception(1)
                break
            except Exception as e:
                print(e)
                print("Invalid choice. Please try again.")
        note = input("Enter any associated notes here: ")
        data.notes = str(note)
        while True:
            try:
                confirm = input("Do you want to add this data? (y/n) ")
                if confirm == "y":
                    self.data.append(data)
                    print("Data has been saved.")
                elif confirm == "n":
                    print("Data not saved.")
                else:
                    raise Exception(1)
                break
            except Exception:
                print("Invalid choice. Please try again.")

    def view(self):
        table = PrettyTable()
        table.field_names = self.defaultFileHeader()
        for row in self.data:
            table.add_row([
            row.tubeLength,
            row.tubeDiameter,
            row.heightDiff,
            row.flowTime,
            round(row.flowVolumeBefore,5),
            round(row.flowVolumeAfter,5),
            ("N/A" if str(row.notes) == "" else str(row.notes))
            ])
        print(table)
        pass

    def plot(self, option):
        fig = plt.figure(option)
        ax = fig.add_subplot(111)
        ax.tick_params(axis="both", direction="in")
        if option == 41:
            group = self.groupBy("tubeDiameter")
            x_data = group.listFromData("tubeDiameter")
            y_data = group.listFromData("avgFlowRate")
            # y_error = group.listFromData("stdErrFlowRate")
            y_error = group.listFromData("uncertaintyFlowRate")
            plt.errorbar(x_data, y_data, yerr=y_error, fmt='ko', markersize=4, elinewidth=1)
            plt.xlabel("Tube diameter (mm)")
            plt.ylabel("Flow Rate (cm^3/s)")
        elif option == 42:
            group = self.groupBy("tubeDiameter")
            x_data = group.listFromData("tubeDiameterP4")
            y_data = group.listFromData("avgFlowRate")
            # y_error = group.listFromData("stdErrFlowRate")
            y_error = group.listFromData("uncertaintyFlowRate")
            plt.errorbar(x_data, y_data, yerr=y_error, fmt='ko', markersize=4, elinewidth=1)
            plt.xlabel("[Tube diameter]^4 (mm^4)")
            plt.ylabel("Flow Rate (cm^3/s)")
        elif option in [43, 44]:
            group = self.groupBy("tubeDiameter")
            x_data = group.listFromData("tubeDiameter")
            y_data = group.listFromData("avgFlowRate")
            # y_error = group.listFromData("stdErrFlowRate")
            y_error = group.listFromData("uncertaintyFlowRate")
            plt.errorbar(x_data, y_data, yerr=y_error, fmt='ko', markersize=4, elinewidth=1)

            y_std = group.listFromData("stdFlowRate")
            xp = np.linspace(min(x_data), max(x_data), 100)
            lFitP = {}
            csq = {}
            tb = PrettyTable()
            tb.field_names = ['Name', 'Poly Fit Equation', 'Chi-squared', 'p-value']
            def fitFunc(p):
                def func(x, a, b):
                    return a + b * x ** p
                return func
            for i in [2, 3, 4]:
                lFitP[i], _ = curve_fit(fitFunc(i), np.array(x_data), np.array(y_data))
                plt.plot(xp, fitFunc(i)(xp, *lFitP[i]), '-', linewidth=1, label=r"$d^{0}$ fit".format(i))
                csq = cst.chisquare(obs=y_data, exp=[fitFunc(i)(xd, *lFitP[i]) for xd in x_data], std=y_error, ddof=2)
                # csq = stats.chisquare(y_data, f_exp=[fitFunc(i)(xd, *lFitP[i]) for xd in x_data], ddof=1)
                dgs = 4
                substrs = []
                for j, k in enumerate(lFitP[i]):
                    if j == 0:
                        apStr = str(round(k, dgs))
                        substrs.append(apStr)
                    elif j == 1:
                        apStr = str(round(k, dgs)) + "x^" + str(i)
                        substrs.append(apStr)
                pStr = "y=" + "+".join(substrs)
                tb.add_row(["d^{0} fit".format(i), pStr, str(round(csq[0],5)), str(round(csq[1],5))])

            print(tb)
            plt.legend(loc="upper left")

            plt.xlabel("Tube diameter (mm)")
            plt.ylabel("Flow Rate (cm^3/s)")
            if option == 43:
                plt.semilogy()

        plt.show()

    def subPlot(self, option):
        length = int(len(self.data))
        while True:
            try:
                subs = input("Enter the range natural indices of the data you want to plot, connected by \"-\", [From 1 to " + str(length) + "]: ")
                subDataIndices = list(map(int, np.linspace(int(subs.split("-")[0]), int(subs.split("-")[1]), num=(abs(int(subs.split("-")[1]) - int(subs.split("-")[0]))+1))))
                print(subDataIndices)
                break
            except ValueError:
                print("Invalid input. Please try again.")

        subData = [self.data[(i - 1)] for i in subDataIndices]
        subDataSet = CplryDataSet()
        subDataSet.data = subData
        subDataSet.plot(option)


    def listFromData(self, option):
        dList = []
        if option == "tubeDiameter":
            dList = [d.tubeDiameter for d in self.data]
        elif option == "tubeDiameterP4":
            dList = [(d.tubeDiameter ** 4) for d in self.data]
        elif option == "avgFlowRate":
            dList = [gd.avgFlowRate() for gd in self.data]
        elif option == "stdErrFlowRate":
            dList = [gd.stdErr() for gd in self.data]
        elif option == "uncertaintyFlowRate":
            dList = [(((self.uncertaintyInV / mean(gd.flowVolumeDiffs)) + (self.uncertaintyInT / mean(gd.flowTimes))) * gd.avgFlowRate()) for gd in self.data]
        elif option == "stdFlowRate":
            dList = [gd.std() for gd in self.data]
        return dList

    def groupBy(self, option):
        groupSet = CplryDataSet()
        tempData = self.data
        groupSet.data = []
        if option == "tubeDiameter":
            paraList = [d.tubeDiameter for d in tempData]
            paraSet = list(set(paraList))
            for p in paraSet:
                pData = CplrySingleGroupData()
                pData.tubeDiameter = p
                pData.groupBy = "tubeDiameter"
                pData.flowRates = []
                for d in tempData:
                    if math.isclose(p, d.tubeDiameter):
                        pData.flowRates.append(d.flowRate())
                        pData.flowTimes.append(d.flowTime)
                        pData.flowVolumeDiffs.append(d.flowVolumeDiff())
                groupSet.data.append(pData)

        return groupSet


def initiate():
    print("----Capillary Lab Data Processing Tool by Jerry Yan----")
    while True:
        opt = input(
            "Choose the following options: \n1 - Create a new data set \n2 - Open an existing data set \nYour choice: ")
        try:
            if opt == "1":
                newDataSet()
            elif opt == "2":
                openDataSet()
            else:
                raise Exception(1)
            break
        except Exception:
            print("Invalid choice. Please try again.")
        else:
            print("Unknown error. Please try again.")

def newDataSet():
    global dataSet
    nm = input("Name the new data set: ")
    dataSet = CplryDataSet(nm)
    print("You created a new data set called " + set.name)

def openDataSet():
    global dataSet
    nm = input("Enter the name of the existing data set: ")
    dataSet = CplryDataSet(nm)
    dataSet.openFrom()

def addOrSave():
    global dataSet
    global passAddOrSave
    while True:
        try:
            opt = input("Choose the following options: \n1 - Add new data\n2 - Print the current data set\n3 - Save the data set\n41 -  Plot tube diameter vs flow rate\n42 -  Plot [tube diameter]^4 vs flow rate\n43 - Plot tube diamter vs flow rate using semilogy() w./ power fits \n44 - Plot tube diamter vs flow rate (Regular plot) w./ power fits \n101 - Plot with a subset\n0 - Exit the program \nYour choice: ")
            if opt == "1":
                dataSet.add()
            elif opt == "2":
                dataSet.view()
            elif opt == "3":
                dataSet.save()
            elif 40 <= int(opt) <= 59:
                dataSet.plot(int(opt))
            elif opt =="101":
                while True:
                    try:
                        pltOpt = input("Please Enter your plotting option: ")
                        pltOpt = int(pltOpt)
                        break
                    except ValueError:
                        print("Invalid Value. Please try again.")
                dataSet.subPlot(pltOpt)
            elif opt == "0":
                while True:
                    try:
                        opt2 = input(
                            "Unsaved data will be lost. Are you sure to continue? (y/n) ")
                        if opt2 == "y":
                            passAddOrSave = True
                        elif opt2 == "n":
                            pass
                        else:
                            raise Exception(1)
                        break
                    except Exception:
                        print("Invalid choice. Please try again.")
            else:
                raise Exception(1)
            break
        except Exception as e:
            traceback.print_exc()
            print(e)
            print("Invalid choice. Please try again.")
        else:
            print("Unknown error.")

passAddOrSave = False

dataSet = CplryDataSet()

initiate()

while passAddOrSave != True:
    addOrSave()

print("Session ended.")
