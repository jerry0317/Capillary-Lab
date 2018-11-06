#!/usr/bin/env python3

# Capillary Lab Data Processing Tool
# Copyright by Jerry Yan, 2018

import csv
import os
import fnmatch
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
    tubeLength = 0.0  # inch
    tubeDiameter = 0.0  # mm
    heightDiff = 0.0  # cm: Height Difference between the top surface and the connected end of the tube
    flowTime = 0.0  # s
    flowVolumeBefore = 0.0  # mL
    flowVolumeAfter = 0.0  # mL
    uncertaintyInT = 0.0  # s
    uncertaintyInV = 0.0  # mL
    notes = ""

    def flowVolumeDiff(self):
        return (self.flowVolumeAfter - self.flowVolumeBefore)

    def tubeRadius(self):
        return (self.tubeDiameter / 2)

    def flowPressure(self):
        return (997 * 9.8 * (self.heightDiff) / 100)

    def flowRate(self):
        return (self.flowVolumeDiff() / self.flowTime)

    def uncertaintyFlowRate(self):
        return (((self.uncertaintyInV / self.flowVolumeDiff()) + (self.uncertaintyInT / self.flowTime)) * self.flowRate())


class CplrySingleGroupData(CplryData):
    groupedBy = ""
    flowRates = []
    flowTimes = []
    flowVolumeDiffs = []
    uncertainties = []

    def __init__(self):
        self.flowRates = []
        self.flowTimes = []
        self.flowVolumeDiffs = []
        self.uncertainties = []

    def avgFlowRate(self):
        return mean(self.flowRates)

    def std(self):
        try:
            std = stdev(self.flowRates)
        except StatisticsError:
            std = self.avgUncertaintyFlowRate()
        return std

    def stdErr(self):
        n = len(self.flowRates)
        stdErr = self.std() / math.sqrt(n)
        return stdErr

    def avgUncertaintyFlowRate(self):
        return mean(self.uncertainties)

    def length(self):
        return len(self.flowRates)


class CplryDataSet:
    name = "Untitled"
    data = []

    def __init__(self, name="Untitled"):
        self.data = []
        self.name = name

    def defaultFilePath(self):
        fileName = self.name + ".csv"
        dirname = os.path.dirname(os.path.abspath(__file__))
        csvFilePath = os.path.join(dirname, "csv", fileName)
        return csvFilePath

    def defaultFileHeader(self):
        return ["Tube Length", "Tube Diameter", "Height Difference", "Flow Time", "Initial Volume", "Final Volume", "Time Uncertainty", "Volume Uncertainty", "Notes"]

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
            dataRow.uncertaintyInT = float(row[h[6]])
            dataRow.uncertaintyInV = float(row[h[7]])
            dataRow.notes = str(row[h[8]])
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
                h[6]: row.uncertaintyInT,
                h[7]: row.uncertaintyInV,
                h[8]: ("N/A" if str(row.notes) == "" else str(row.notes))
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
            try:
                uftTime = input(
                    "Enter the uncertainty of time measurement in s (0.005 by default): ")
                if uftTime == "":
                    data.uncertaintyInT = 0.005
                else:
                    data.uncertaintyInT = float(uftTime)
                break
            except ValueError:
                print("Invalid input. Please try again.")
        while True:
            opt = input(
                "Choose from the following options:\n1 - Enter flow volumes directly\n2 - Enter flow masses to get volumes\nYour choice: ")
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
                    while True:
                        try:
                            utv = input(
                                "Enter the uncertainty in volume measurement in mL (1.0 by default): ")
                            if utv == "":
                                data.uncertaintyInV = 1.0
                            else:
                                data.uncertaintyInV = float(utv)
                            break
                        except ValueError:
                            print("Invalid input. Please try again.")
                elif opt == "2":
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
                            data.flowVolumeAfter = (flowMassAfter / 0.997)
                            break
                        except ValueError:
                            print("Invalid input. Please try again.")
                    while True:
                        try:
                            utm = input(
                                "Enter the uncertainty in mass measurement in g (1.0 by default): ")
                            if utm == "":
                                data.uncertaintyInV = (1 / 0.997)
                            else:
                                utmv = float(utm)
                                data.uncertaintyInV = (utmv / 0.997)
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

    def view(self, option=1):
        table = PrettyTable()
        if option == 1:
            table.field_names = self.defaultFileHeader() + ["Flow Rate", "Uncertainty"]
            for row in self.data:
                table.add_row([
                    row.tubeLength,
                    row.tubeDiameter,
                    row.heightDiff,
                    row.flowTime,
                    round(row.flowVolumeBefore, 5),
                    round(row.flowVolumeAfter, 5),
                    row.uncertaintyInT,
                    round(row.uncertaintyInV, 3),
                    ("N/A" if str(row.notes) == "" else str(row.notes)),
                    round(row.flowRate(), 5),
                    round(row.uncertaintyFlowRate(), 5)
                ])
        elif option == 2:
            table.field_names = ["Tube Diameter", "Avg Flow Rate", "Uncertainty", "STD"]
            for row in self.data:
                table.add_row([
                    row.tubeDiameter,
                    round(row.avgFlowRate(), 5),
                    round(row.avgUncertaintyFlowRate(), 5),
                    round(row.std(), 5)
                ])
        elif option == 3:
            table.field_names = ["Tube Length", "Avg Flow Rate", "Uncertainty", "STD"]
            for row in self.data:
                table.add_row([
                    row.tubeLength,
                    round(row.avgFlowRate(), 5),
                    round(row.avgUncertaintyFlowRate(), 5),
                    round(row.std(), 5)
                ])
        elif option == 4:
            table.field_names = ["Height Difference", "flowPressure", "Avg Flow Rate", "Uncertainty", "STD"]
            for row in self.data:
                table.add_row([
                    row.heightDiff,
                    round(row.flowPressure(), 3),
                    round(row.avgFlowRate(), 5),
                    round(row.avgUncertaintyFlowRate(), 5),
                    round(row.std(), 5)
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
            y_error = group.listFromData("uErrorFlowRate")
            plt.errorbar(x_data, y_data, yerr=y_error,
                         fmt='ko', markersize=4, elinewidth=1)
            plt.xlabel("Tube diameter (mm)")
            plt.ylabel("Flow Rate (cm^3/s)")
        elif option == 42:
            group = self.groupBy("tubeDiameter")
            x_data = group.listFromData("tubeDiameterP4")
            y_data = group.listFromData("avgFlowRate")
            y_error = group.listFromData("uErrorFlowRate")
            plt.errorbar(x_data, y_data, yerr=y_error,
                         fmt='ko', markersize=4, elinewidth=1)
            plt.xlabel("[Tube diameter]^4 (mm^4)")
            plt.ylabel("Flow Rate (cm^3/s)")
        elif option in [43, 44]:
            group = self.groupBy("tubeDiameter")
            x_data = group.listFromData("tubeDiameter")
            y_data = group.listFromData("avgFlowRate")
            y_error = group.listFromData("uErrorFlowRate")
            plt.errorbar(x_data, y_data, yerr=y_error, fmt='ko', markersize=4, elinewidth=1)

            y_sigma = group.listFromData("uncertaintyFlowRate")
            xp = np.linspace(min(x_data), max(x_data), 100)
            lFitP = {}
            tb = PrettyTable()
            tb.field_names = ['Name', 'Poly Fit Equation', 'Chi-squared', 'p-value']
            for i in [2, 3, 4, 5]:
                lFitP[i], csq, eq, func = CplryDataSet.singlePloyFit(x_data, y_data, y_sigma, i, 4)
                plt.plot(xp, func(xp, *lFitP[i]), '-', linewidth=1, label=r"$d^{0}$ fit".format(i))
                tb.add_row(["d^{0} fit".format(i), eq, str(round(csq[0], 5)), str(round(csq[1], 5))])

            print(tb)
            plt.legend(loc="upper right")

            plt.xlabel("Tube diameter (mm)")
            plt.ylabel("Flow Rate (cm^3/s)")
            if option == 43:
                plt.semilogy()

        elif option in [51, 52]:
            group = self.groupBy("tubeLength")
            x_data = group.listFromData("tubeLength")
            y_data = group.listFromData("avgFlowRate")
            if option == 51:
                y_error = group.listFromData("uErrorFlowRate")
            elif option == 52:
                y_error = group.listFromData("stdErrFlowRate")
            plt.errorbar(x_data, y_data, yerr=y_error, fmt="ko", markersize=4, elinewidth=1)

            if option == 51:
                y_sigma = group.listFromData("uncertaintyFlowRate")
            elif option == 52:
                y_sigma = group.listFromData("stdFlowRate")
            xp = np.linspace(min(x_data), max(x_data), 100)
            tb = PrettyTable()
            tb.field_names = ['Model Name', 'Fit Equation', 'Chi-squared', 'p-value']
            for i in [-2, -1, -0.5]:
                lFitP, csq, eq, func = CplryDataSet.singlePloyFit(x_data, y_data, y_sigma, i, 4)
                plt.plot(xp, func(xp, *lFitP), '-', linewidth=1, label=r"l^{0} fit".format(i))
                tb.add_row(["l^{0} fit".format(i), eq, str(round(csq[0], 5)), str(round(csq[1], 5))])
            print(tb)
            plt.legend(loc="upper left")

            plt.xlabel("Tube length (inch)")
            plt.ylabel("Flow Rate (cm^3/s)")

        elif option == 61:
            group = self.groupBy("heightDiff")
            x_data = group.listFromData("flowPressure")
            y_data = group.listFromData("avgFlowRate")
            y_error = group.listFromData("uErrorFlowRate")
            plt.error(x_data, y_data, yerr=y_error, fmt="ko", markersize=4, elinewidth=1)

            y_sigma = group.listFromData("uncertaintyFlowRate")
            xp = np.linspace(min(x_data), max(x_data), 100)
            tb = PrettyTable()
            tb.field_names = ['Model Name', 'Fit Equation', 'Chi-squared', 'p-value']
            for i in [0.5, 1, 2]:
                lFitP, csq, eq, func = CplryDataSet.singlePloyFit(x_data, y_data, y_sigma, i, 4)
                plt.plot(xp, func(xp, *lFitP), '-', linewidth=1, label=r"l^{0} fit".format(i))
                tb.add_row(["P^{0} fit".format(i), eq, str(round(csq[0], 5)), str(round(csq[1], 5))])
            print(tb)

            plt.legend(loc="upper left")

            plt.xlabel("Pressure (Pa)")
            plt.ylabel("Flow Rate (cm^3/s)")

        plt.show()

    def subPlot(self, option):
        length = int(len(self.data))
        while True:
            try:
                subs = input("Enter the range natural indices of the data you want to plot, connected by \"-\", [From 1 to " + str(length) + "]: ")
                subDataIndices = list(map(int, np.linspace(int(subs.split("-")[0]), int(subs.split("-")[1]), num=(abs(int(subs.split("-")[1]) - int(subs.split("-")[0])) + 1))))
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
        elif option == "tubeLength":
            dList = [d.tubeLength for d in self.data]
        elif option == "flowPressure":
            dList = [d.flowPressure() for d in self.data]
        elif option == "avgFlowRate":
            dList = [gd.avgFlowRate() for gd in self.data]
        elif option == "stdErrFlowRate":
            dList = [gd.stdErr() for gd in self.data]
        elif option == "uncertaintyFlowRate":
            dList = [gd.avgUncertaintyFlowRate() for gd in self.data]
        elif option == "uErrorFlowRate":
            dList = [gd.avgUncertaintyFlowRate() / math.sqrt(gd.length())
                     for gd in self.data]
        elif option == "stdFlowRate":
            dList = [gd.std() for gd in self.data]
        return dList

    def groupBy(self, option):
        groupSet = CplryDataSet()
        tempData = self.data
        groupSet.data = []

        def gParameter(dt):
            if option == "tubeDiameter":
                return dt.tubeDiameter
            elif option == "tubeLength":
                return dt.tubeLength
            elif option == "heightDiff":
                return dt.heightDiff
            else:
                return None

        def assignGP(a, b):
            if option == "tubeDiameter":
                a.tubeDiameter = b
            elif option == "tubeLength":
                a.tubeLength = b
            elif option == "heightDiff":
                a.heightDiff = b

        paraList = [gParameter(d) for d in tempData]
        paraSet = list(set(paraList))
        for p in paraSet:
            pData = CplrySingleGroupData()
            assignGP(pData, p)
            pData.groupBy = option
            for d in tempData:
                if math.isclose(p, gParameter(d)):
                    pData.flowRates.append(d.flowRate())
                    pData.flowTimes.append(d.flowTime)
                    pData.flowVolumeDiffs.append(d.flowVolumeDiff())
                    pData.uncertainties.append(d.uncertaintyFlowRate())
            groupSet.data.append(pData)
        groupSet.data.sort(key=lambda d: gParameter(d))

        return groupSet

    def calculate(self, option):
        if option == 121:
            while True:
                try:
                    sigLI = input(
                        "Enter the desired significance level (0.05 by default): ")
                    if sigLI == "":
                        sigL = 0.05
                    else:
                        sigL = float(sigLI)
                    if (sigL > 1) or (sigL < 0):
                        raise Exception(3)
                    break
                except Exception(3):
                    print(
                        "Significance level must be between 0 and 1. Please try again.")
                else:
                    traceback.print_exc()

            group = self.groupBy("tubeDiameter")
            group.data.sort(key=lambda d: d.tubeDiameter)

            validList = []

            # Each util must contain at least 3 data points
            lrg = range(3, len(group.data) + 1)
            for lr in lrg:
                irg = range(0, len(group.data) - lr + 1)  # The index range
                for ir in irg:
                    subGroup = group.data[ir: ir + lr]
                    subGroupSet = CplryDataSet()
                    subGroupSet.data = subGroup
                    x_data = subGroupSet.listFromData("tubeDiameter")
                    y_data = subGroupSet.listFromData("avgFlowRate")
                    y_error = subGroupSet.listFromData("uncertaintyFlowRate")
                    for i in [2, 3, 4, 5]:
                        lFitP, csq, eq, _ = CplryDataSet.singlePloyFit(
                            x_data, y_data, y_error, i, 4)

                        if csq[1] >= sigL:
                            vData = {
                                "dataSet": subGroupSet,
                                "power": i,
                                "fitModel": lFitP,
                                "csq": csq,
                                "equation": eq,
                                "chi-squared": csq[0],
                                "p-value": csq[1]
                            }
                            validList.append(vData)

            tb = PrettyTable()
            tb.field_names = ["Data Set", "Data Length",
                              "Model Type", "Fit Equation", "Chi-sqaured", "p-value"]
            for ld in validList:
                dataSetStr = ", ".join(
                    [str(round(d.tubeDiameter, 4)) + "mm" for d in ld["dataSet"].data])
                tb.add_row([dataSetStr, len(ld["dataSet"].data), "d^{0} fit".format(
                    ld["power"]), ld["equation"], round(ld["chi-squared"], 4), round(ld["p-value"], 4)])
            print(tb)

    @staticmethod
    def singlePloyFit(xd, yd, yerr, power, dgs=4):
        def fitFunc(p):
            def func(x, a, b):
                return a + b * x ** p
            return func
        lFitP, _ = curve_fit(fitFunc(power), np.array(
            xd), np.array(yd), sigma=yerr, absolute_sigma=True)
        csq = cst.chisquare(obs=yd, exp=[fitFunc(power)(
            xpt, *lFitP) for xpt in xd], std=yerr, ddof=2)

        substrs = []
        for j, k in enumerate(lFitP):
            if j == 0:
                apStr = str(round(k, dgs))
                substrs.append(apStr)
            elif j == 1:
                apStr = str(round(k, dgs)) + "x^" + str(power)
                substrs.append(apStr)
        pStr = "y=" + "+".join(substrs)

        return lFitP, csq, pStr, fitFunc(power)


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
            traceback.print_exc()
            print("Invalid choice. Please try again.")
        else:
            print("Unknown error. Please try again.")


def newDataSet():
    global dataSet
    nm = input("Name the new data set: ")
    dataSet = CplryDataSet(nm)
    print("You created a new data set called " + dataSet.name)


def openDataSet():
    global dataSet
    flsList = []
    dirname = os.path.dirname(os.path.abspath(__file__))
    csvPath = os.path.join(dirname, "csv")
    for root, dirs, files in os.walk(csvPath):
        for filename in fnmatch.filter(files, "*.csv"):
            flsList.append(os.path.splitext(filename)[0])
    instrs = []
    for i in range(0, len(flsList)):
        substr = "{0} - {1}".format(i + 1, flsList[i])
        instrs.append(substr)
    instr = "Choose from the following files:\n" + \
        "\n".join(instrs) + "\nYour choice: "
    while True:
        try:
            opt = input(instr)
            if (int(opt) - 1) not in range(0, len(flsList)):
                raise Exception(2)
            else:
                dataSet = CplryDataSet(flsList[int(opt) - 1])
            break
        except Exception as e:
            traceback.print_exc()
            print(e)
            print("Invalid choice. Please try again.")
        else:
            print("Unknown error.")

    dataSet.openFrom()


def addOrSave():
    global dataSet
    global passAddOrSave
    while True:
        try:
            opt = input(
                "-----Current Set Name: {0}-----\nChoose the following options: \n1 - Add new data\n2 - Print the current data set\n21 - Print the data set w./ group by tube diameter\n22 - Print the data set w./ group by tube length\n23 - Print the data set w./ group by height diff\n3 - Save the data set\n41 - Plot tube diameter vs flow rate\n42 - Plot [tube diameter]^4 vs flow rate\n43 - Plot tube diamter vs flow rate using semilogy() w./ power fits \n44 - Plot tube diamter vs flow rate (Regular plot) w./ power fits\n51 - Plot tube length vs flow rate w./ inverse poly fit using uncertainty\n52 - Plot tube length vs flow rate w./ inverse poly fit using STD\n61 - Plot flow pressure vs flow rate w./ inverse poly fit using uncertainty \n101 - Plot with a subset\n121 - Find fit model about tube diameter vs flow rate with significance level\n9 - Switch to another data set\n0 - Exit the program \nYour choice: ".format(dataSet.name))
            if opt == "1":
                dataSet.add()
            elif opt == "2":
                dataSet.view()
            elif opt == "21":
                group = dataSet.groupBy("tubeDiameter")
                group.view(2)
            elif opt == "22":
                group = dataSet.groupBy("tubeLength")
                group.view(3)
            elif opt == "23":
                group = dataSet.groupBy("heightDiff")
                group.view(4)
            elif opt == "3":
                dataSet.save()
            elif 40 <= int(opt) <= 69:
                dataSet.plot(int(opt))
            elif opt == "9":
                openDataSet()
            elif opt == "101":
                while True:
                    try:
                        pltOpt = input("Please Enter your plotting option: ")
                        pltOpt = int(pltOpt)
                        break
                    except ValueError:
                        print("Invalid Value. Please try again.")
                dataSet.subPlot(pltOpt)
            elif 120 <= int(opt) <= 139:
                dataSet.calculate(int(opt))
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

print("-----Session ended.-----")
