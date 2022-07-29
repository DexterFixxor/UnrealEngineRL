import numpy as np

class Robot:

    def __init__(self, name, controllerNameList, jointUpperLimits, jointLowerLimits):

        self.name = name
        self.controllerNameList = controllerNameList
        self.jointUpperLimits = jointUpperLimits
        self.jointLowerLimits = jointLowerLimits

        self.range = (np.array(self.jointUpperLimits) - np.array(self.jointLowerLimits))/2
        self.offset = (np.array(self.jointUpperLimits) + np.array(self.jointLowerLimits))/2

    def actionDict(self, actions):
        ret = {
            "robot":
                {
                    "name": self.name,
                    "joints": self.controllerNameList,
                    "actions": list(actions * self.range + self.offset)
                }
        }

        return ret

    def unscaledActionDict(self, actions):
        ret = {
            "robot":
                {
                    "name": self.name,
                    "joints": self.controllerNameList,
                    "actions": list(actions)
                }
        }

        return ret


    def __str__(self):
        return f"Robot:\n" \
               f"\tName: {self.name}\n" \
               f"\tControllers: {self.controllerNameList}\n" \
               f"\tRanges: {self.range}\n" \
               f"\tOffsets: {self.offset}"




    

