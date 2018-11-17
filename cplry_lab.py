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
        return (997 * 9.8 * (self.heightDiff) / 100) # Pa

    def effectivePressure(self):
        return (self.flowPressure() + (997 * 9.8 * (self.tubeLength) * 2.54 / 100)) # Pa

    def flowRate(self):
        return (self.flowVolumeDiff() / self.flowTime)

    def uncertaintyFlowRate(self):
        return (((self.uncertaintyInV / self.flowVolumeDiff()) + (self.uncertaintyInT / self.flowTime)) * self.flowRate())

    def reynoldsNumber(self):
        rsn = 4 * (self.flowRate()*10**(-6)) * (997) / (0.9775*10**(-3)*math.pi*(self.tubeDiameter*10**(-3)))
        return rsn


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
    __mu = 0.9775 * (10 ** (-3))

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

    def __promptForValue(self, prompt, default=None):
        while True:
            try:
                if default != None:
                    prompt = prompt + " ({} by default): ".format(default)
                else:
                    prompt = prompt + ": "
                value = input(prompt)
                if default != None and value == "":
                    return default
                return float(value)
                break
            except ValueError:
                print("Invalid input. Please try again.")

    def add(self, option=0):
        print("You initiated a new row of data.")
        data = CplryData()
        data.tubeLength = self.__promptForValue("Enter the tube length in inch")
        data.tubeDiameter = self.__promptForValue("Enter the tube diamter in mm")
        data.heightDiff = self.__promptForValue("Enter the height difference in cm")
        data.flowTime = self.__promptForValue("Enter the flow time in s")
        data.uncertaintyInT = self.__promptForValue("Enter the uncertainty of time measurement in s", 0.005)
        while True:
            opt = input(
                "Choose from the following options:\n1 - Enter flow volumes directly\n2 - Enter flow masses to get volumes\nYour choice: ")
            try:
                if opt == "1":
                    data.flowVolumeBefore = self.__promptForValue("Enter the initial volume in mL")
                    data.flowVolumeAfter = self.__promptForValue("Enter the final volume in mL")
                    data.uncertaintyInV = self.__promptForValue("Enter the uncertainty in volume measurement in mL", 1.0)
                elif opt == "2":
                    data.flowVolumeBefore = self.__promptForValue("Enter the initial mass in g") / 0.997
                    data.flowVolumeAfter = self.__promptForValue("Enter the final mass in g") / 0.997
                    data.uncertaintyInV = self.__promptForValue("Enter the uncertainty in mass measurement in g", 1.0) / 0.997
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
            table.field_names = self.defaultFileHeader() + ["Flow Rate", "Uncertainty", "Re"]
            for row in self.data:
                table.add_row([
                    row.tubeLength,
                    row.tubeDiameter,
                    row.heightDiff,
                    row.flowTime,
                    round(row.flowVolumeBefore, 1),
                    round(row.flowVolumeAfter, 1),
                    row.uncertaintyInT,
                    round(row.uncertaintyInV, 3),
                    ("N/A" if str(row.notes) == "" else str(row.notes)),
                    round(row.flowRate(), 5),
                    round(row.uncertaintyFlowRate(), 5),
                    round(row.reynoldsNumber(), 3)
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
        elif option == 5:
            table.field_names = ["Shared Label", "Avg Flow Rate", "Uncertainty", "STD"]
            for row in self.data:
                table.add_row([
                    row.notes,
                    round(row.avgFlowRate(), 5),
                    round(row.avgUncertaintyFlowRate(), 5),
                    round(row.std(), 5)
                ])
        print(table)
        pass

    def plot(self, option):
        fig = plt.figure(option)
        ax = fig.add_subplot(111)
        ax.tick_params(axis="both",which="both", direction="in")
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
            tb.field_names = ['Name', 'Poly Fit Equation', 'Theoretical Coeff', 'Chi-squared', 'p-value']
            for i in [4, (19/7)]:
                lFitP[i], csq, eq, func = CplryDataSet.singlePloyFit(x_data, y_data, y_sigma, i, 4)
                plt.plot(xp, func(xp, *lFitP[i]), '-', linewidth=1, label=r"$d^{{{0}}}$ fit".format(round(i,2)))

                effPressure = mean(group.listFromData("effectivePressure"))
                tubeLen = mean(group.listFromData("tubeLength"))
                vis = self.__mu

                theoryCoeff = CplryDataSet.theoryCoeffForD(i, effP=effPressure, mu=vis, tubeL=tubeLen)

                tb.add_row(["d^{0} fit".format(i), eq, str(round(theoryCoeff,5)), str(round(csq[0], 5)), str(round(csq[1], 5))])

            print(tb)
            plt.legend(loc="upper left")

            plt.xlabel("Tube diameter (mm)")
            plt.ylabel(r"Flow Rate (cm$^3$/s)")
            if option == 43:
                plt.semilogy()

        elif option in [51, 52]:
            group = self.groupBy("tubeLength")
            x_data = group.listFromData("tubeLengthCM")
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
            tb.field_names = ['Model Name', 'Fit Equation', 'Theoretical Coeff x^0', 'Theoretical Coeff x^i', 'Chi-squared', 'p-value']
            for i in [-1, -4/7]:
                lFitP, csq, eq, func = CplryDataSet.singlePloyFit(x_data, y_data, y_sigma, i, 4)
                plt.plot(xp, func(xp, *lFitP), '-', linewidth=1, label=r"$l^{{{0}}}$ fit".format(round(i,2)))

                hPressure = mean(group.listFromData("flowPressure"))
                tubeDiameter = mean(group.listFromData("tubeDiameter"))
                vis = self.__mu

                theoryCoeff = CplryDataSet.theoryCoeffForLCM(i, hP=hPressure, tubeD=tubeDiameter, mu=vis)

                tb.add_row(["l^{0} fit".format(round(i,2)), eq, str(round(theoryCoeff[0],5)), str(round(theoryCoeff[1],5)), str(round(csq[0], 5)), str(round(csq[1], 5))])
            print(tb)
            plt.legend(loc="upper right")

            plt.xlabel("Tube length (cm)")
            plt.ylabel(r"Flow Rate (cm$^3$/s)")

        elif option in [61, 62, 63, 64]:
            group = self.groupBy("heightDiff")
            if option in [61, 62]:
                x_data = [d / 1000 for d in group.listFromData("flowPressure")]
                x_label = "Pressure (kPa)"
            elif option in [63, 64]:
                x_data = [d / 1000 for d in group.listFromData("effectivePressure")]
                x_label = "Effective Pressure (kPa)"

            y_data = group.listFromData("avgFlowRate")
            x_error = 997 * 9.8 * (0.5) / (100 * 1000)
            if option in [61, 63]:
                y_error = group.listFromData("uErrorFlowRate")
                y_sigma = group.listFromData("uncertaintyFlowRate")

            elif option in [62, 64]:
                y_error = group.listFromData("stdErrFlowRate")
                y_sigma = group.listFromData("stdFlowRate")

            plt.errorbar(x_data, y_data, xerr=x_error, yerr=y_error, fmt="ko", markersize=4, elinewidth=1)

            xp = np.linspace(min(x_data), max(x_data), 100)
            tb = PrettyTable()
            tb.field_names = ['Model Name', 'Fit Equation', 'Theoretical Coeff (P_eff)', 'Chi-squared', 'p-value']
            for i in [1, 4/7]:
                lFitP, csq, eq, func = CplryDataSet.singlePloyFit(x_data, y_data, y_sigma, i, 4)
                plt.plot(xp, func(xp, *lFitP), '-', linewidth=1, label=r"$P^{{{0}}}$ fit".format(round(i,2)))

                tubeLength = mean(group.listFromData("tubeLength"))
                tubeDiameter = mean(group.listFromData("tubeDiameter"))
                vis = self.__mu


                theoryCoeff = CplryDataSet.theoryCoeffForEffP(i, tubeD=tubeDiameter, tubeL=tubeLength, mu=vis)

                tb.add_row(["P^{0} fit".format(round(i,2)), eq, str(round(theoryCoeff,5)), str(round(csq[0], 5)), str(round(csq[1], 5))])
            print(tb)

            plt.legend(loc="upper left")

            plt.xlabel(x_label)
            plt.ylabel(r"Flow Rate (cm$^3$/s)")

        elif option in [45, 46]:
            group = self.groupBy("tubeDiameter")
            x_data = group.listFromData("tubeDiameter")
            y_data = group.listFromData("avgFlowRate")
            y_error = group.listFromData("uErrorFlowRate")
            plt.errorbar(x_data, y_data, yerr=y_error, fmt='ko', markersize=4, elinewidth=1)

            calList = self.calculate(121, True)

            i = 1
            for c in calList:
                gx = c["dataSet"].listFromData("tubeDiameter")
                gxp = np.linspace(min(gx), max(gx), 100)
                glFitP = c["fitModel"]
                gFunc = c["fitFunc"]
                plt.plot(gxp, gFunc(gxp, *glFitP), '-', linewidth=1, label=r"$L_{0}$".format(i))
                i += 1

            plt.legend(loc="upper left")

            plt.xlabel("Tube diameter (mm)")
            plt.ylabel(r"Flow Rate (cm$^3$/s)")
            if option == 45:
                plt.semilogy()

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
        elif option == "tubeLengthCM":
            dList = [d.tubeLength * 2.54 for d in self.data]
        elif option == "flowPressure":
            dList = [d.flowPressure() for d in self.data]
        elif option == "effectivePressure":
            dList = [d.effectivePressure() for d in self.data]
        elif option == "avgFlowRate":
            dList = [gd.avgFlowRate() for gd in self.data]
        elif option == "stdErrFlowRate":
            dList = [gd.stdErr() for gd in self.data]
        elif option == "uncertaintyFlowRate":
            dList = [gd.avgUncertaintyFlowRate() for gd in self.data]
        elif option == "uErrorFlowRate":
            dList = [gd.avgUncertaintyFlowRate() / math.sqrt(gd.length()) for gd in self.data]
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
            elif option == "notes":
                return dt.notes
            else:
                return None

        paraList = [gParameter(d) for d in tempData]
        paraSet = list(set(paraList))
        for p in paraSet:
            pData = CplrySingleGroupData()
            # assignGP(pData, p)
            pData.groupBy = option
            for d in tempData:
                if (math.isclose(p, gParameter(d)) if type(gParameter(d)) != str else p == gParameter(d)) :
                    pData.flowRates.append(d.flowRate())
                    pData.flowTimes.append(d.flowTime)
                    pData.flowVolumeDiffs.append(d.flowVolumeDiff())
                    pData.uncertainties.append(d.uncertaintyFlowRate())
                    pData.tubeDiameter = d.tubeDiameter
                    pData.tubeLength = d.tubeLength
                    pData.heightDiff = d.heightDiff
                    pData.notes = d.notes
            groupSet.data.append(pData)
        groupSet.data.sort(key=lambda d: gParameter(d))

        return groupSet

    def calculate(self, option, ret=False):
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
                    print("Significance level must be between 0 and 1. Please try again.")
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
                    for i in [4, (19/7)]:
                        lFitP, csq, eq, fitFunc = CplryDataSet.singlePloyFit(x_data, y_data, y_error, i, 4)
                        effPressure = mean(subGroupSet.listFromData("effectivePressure"))
                        tubeLen = mean(subGroupSet.listFromData("tubeLength"))
                        vis = self.__mu

                        theoryCoeff = CplryDataSet.theoryCoeffForD(i, effP=effPressure, mu=vis, tubeL=tubeLen)

                        if csq[1] >= sigL:
                            vData = {
                                "dataSet": subGroupSet,
                                "power": i,
                                "fitModel": lFitP,
                                "fitFunc": fitFunc,
                                "csq": csq,
                                "equation": eq,
                                "chi-squared": csq[0],
                                "p-value": csq[1],
                                "theoryCoeff": theoryCoeff
                            }
                            validList.append(vData)

            tb = PrettyTable()
            tb.field_names = ["Data Set", "Data Length", "Model Type", "Fit Equation", "Theoretical Coeff", "Chi-sqaured", "p-value"]
            for ld in validList:
                dataSetStr = ", ".join([str(round(d.tubeDiameter, 4)) + "mm" for d in ld["dataSet"].data])
                tb.add_row([dataSetStr, len(ld["dataSet"].data), "d^{0} fit".format(round(ld["power"],2)), ld["equation"], round(ld["theoryCoeff"], 5), round(ld["chi-squared"], 4), round(ld["p-value"], 4)])
            print(tb)
            if ret == True:
                return validList

        elif option in [131, 132]:
            try:
                alp = self.__promptForValue("Enter the significance level", 0.05)
                group = self.groupBy("notes")
                if len(group.data) != 2:
                    raise Exception(1311)

                r1 = group.data[0].flowRates
                r2 = group.data[1].flowRates

                if option == 131:
                    s = cst.meanDiffInterval(r1, r2, alp, False)
                elif option == 132:
                    x_u = group.data[0].avgUncertaintyFlowRate()
                    y_u = group.data[1].avgUncertaintyFlowRate()
                    s = cst.meanDiffInterval(r1, r2, alp, False, x_u, y_u)

                print("RESULT: The estimate mean difference is {0}, and the {1} %% confidence interval is ({2}, {3})".format(round(s[0],5), round((1-alp)*100, 1), round(s[1],5), round(s[2],5)))

            except Exception(1311):
                print("This data set is not valid for this operation.")



    @staticmethod
    def singlePloyFit(xd, yd, yerr, power, dgs=4):
        def fitFunc(p):
            def func(x, a, b):
                return a + b * x ** p
            return func
        lFitP, _ = curve_fit(fitFunc(power), np.array(xd), np.array(yd), sigma=yerr, absolute_sigma=True)
        csq = cst.chisquare(obs=yd, exp=[fitFunc(power)(xpt, *lFitP) for xpt in xd], std=yerr, ddof=2)

        substrs = []
        for j, k in enumerate(lFitP):
            if j == 0:
                apStr = str(round(k, dgs))
                substrs.append(apStr)
            elif j == 1:
                apStr = str(round(k, dgs)) + "x^" + str(round(power,2))
                substrs.append(apStr)
        pStr = "y=" + "+".join(substrs)

        return lFitP, csq, pStr, fitFunc(power)

    @staticmethod
    def theoryCoeffForD(i, effP, tubeL, mu):
        if i == 4: # Laminar model
            tC = effP * math.pi / (128 * mu * tubeL * 2.54 / 100) * (10 ** (-6))
        elif math.isclose(i, 19/7): # Approx turbulent model
            tC = 2.255 * math.pow(effP, 4/7) / (math.pow(997, 3/7) * math.pow(mu, 1/7) * math.pow(tubeL * 2.54 / 100, 4/7)) * math.pow(10, -15/7)
        return tC

    @staticmethod
    def theoryCoeffForLCM(i, hP, tubeD, mu):
        if i == -1: # Laminar model
            tC1 = hP * math.pi * math.pow(tubeD/1000, 4) / (128 * mu) * (10 ** 8)
            tC0 = 997 * 9.8 * math.pi * math.pow(tubeD/1000, 4) / (128 * mu) * (10 ** 6)
        elif math.isclose(i, -4/7): # Approx turbulent model [HORIZONTAL ONLY]
            tC1 = 2.255 * math.pow(hP, 4/7) * math.pow(tubeD/1000, 19/7) / (math.pow(997, 3/7) * math.pow(mu, 1/7)) * math.pow(10, 50/7)
            tC0 = 0
        return [tC0, tC1]

    @staticmethod
    def theoryCoeffForEffP(i, tubeD, tubeL, mu, scale=1000):
        if i == 1: # Laminar model
            tC = math.pi * math.pow(tubeD/1000, 4) / ((128 * mu) * (tubeL * 2.54 / 100)) * (10 ** 6) * scale
        elif math.isclose(i, 4/7): # Approx turbulent model
            tC = 2.255 * math.pow(tubeD/1000, 19/7) / (math.pow(997, 3/7) * math.pow(mu, 1/7) * math.pow(tubeL * 2.54 / 100, 4/7)) * (10 ** 6) * math.pow(scale, 4/7)
        return tC



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
    flsList.sort()
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
            opt = input("""-----Current Set Name: {0}-----
Choose the following options:
1 - Add new data
2 - Print the current data set
21 - Print the data set w./ group by tube diameter
22 - Print the data set w./ group by tube length
23 - Print the data set w./ group by height diff
24 - Print the data set w./ group by notes
3 - Save the data set
41 - Plot tube diameter vs flow rate
42 - Plot [tube diameter]^4 vs flow rate
43 - Plot tube diamter vs flow rate using semilogy() w./ power fits
44 - Plot tube diamter vs flow rate (Regular plot) w./ power fits
45 - Use 121 & plot tube diamter vs flow rate using semilogy() w./ power fits
46 - Use 121 & plot tube diamter vs flow rate (Regular plot) w./ power fits
51 - Plot tube length vs flow rate w./ inverse poly fit using uncertainty
52 - Plot tube length vs flow rate w./ inverse poly fit using STD
61 - Plot flow pressure vs flow rate w./ power fits using uncertainty
62 - Plot flow pressure vs flow rate w./ power fits using STD
63 - Plot flow pressure [effective] vs flow rate w./ power fits using uncertainty
64 - Plot flow pressure [effective] vs flow rate w./ power fits using STD
101 - Plot with a subset
121 - Find fit model about tube diameter vs flow rate with significance level
131 - Find the confidence interval of the estimated mean between two setups w./ STD
132 - Find the confidence interval of the estimated mean between two setups w./ uncertainties
9 - Switch to another data set
0 - Exit the program
Your choice: """.format(dataSet.name))
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
            elif opt == "24":
                group = dataSet.groupBy("notes")
                group.view(5)
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
