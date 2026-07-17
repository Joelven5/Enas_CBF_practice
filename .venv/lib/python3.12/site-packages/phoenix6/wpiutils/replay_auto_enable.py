"""
Copyright (C) Cross The Road Electronics.  All rights reserved.
License information can be found in CTRE_LICENSE.txt
For support and suggestions contact support@ctr-electronics.com or file
an issue tracker at https://github.com/CrossTheRoadElec/Phoenix-Releases
"""

try:
    from hal import AllianceStationID
    from wpilib import Notifier
    from wpilib.simulation import DriverStationSim
    from threading import RLock
    from typing import final
    from phoenix6 import HootReplay

    @final
    class ReplayAutoEnable:
        __instance = None

        def __init__(self):
            self.__enable_notifier = Notifier(self.__run)
            self.__lock = RLock()
            self.__start_count = 0

        def __run(self):
            if HootReplay.is_playing():
                ds_attached_sig = HootReplay.get_boolean("DS:IsDSAttached")
                if ds_attached_sig.status.is_ok():
                    DriverStationSim.setDsAttached(ds_attached_sig.value)

                if DriverStationSim.getDsAttached():
                    fms_attached_sig = HootReplay.get_boolean("DS:IsFMSAttached")
                    if fms_attached_sig.status.is_ok():
                        DriverStationSim.setFmsAttached(fms_attached_sig.value)

                    enable_sig = HootReplay.get_boolean("RobotEnable")
                    if enable_sig.status.is_ok():
                        DriverStationSim.setEnabled(enable_sig.value)

                    robot_mode_sig = HootReplay.get_string("RobotMode")
                    if robot_mode_sig.status.is_ok():
                        if robot_mode_sig.value == "Autonomous":
                            DriverStationSim.setAutonomous(True)
                            DriverStationSim.setTest(False)
                            DriverStationSim.setEStop(False)
                        elif robot_mode_sig.value == "Test":
                            DriverStationSim.setAutonomous(False)
                            DriverStationSim.setTest(True)
                            DriverStationSim.setEStop(False)
                        elif robot_mode_sig.value == "EStopped":
                            DriverStationSim.setAutonomous(False)
                            DriverStationSim.setTest(False)
                            DriverStationSim.setEStop(True)
                        else:
                            DriverStationSim.setAutonomous(False)
                            DriverStationSim.setTest(False)
                            DriverStationSim.setEStop(False)

                    alliance_station_sig = HootReplay.get_string("AllianceStation")
                    if alliance_station_sig.status.is_ok():
                        if alliance_station_sig.value == "Red 1":
                            DriverStationSim.setAllianceStationId(AllianceStationID.kRed1)
                        elif alliance_station_sig.value == "Red 2":
                            DriverStationSim.setAllianceStationId(AllianceStationID.kRed2)
                        elif alliance_station_sig.value == "Red 3":
                            DriverStationSim.setAllianceStationId(AllianceStationID.kRed3)
                        elif alliance_station_sig.value == "Blue 1":
                            DriverStationSim.setAllianceStationId(AllianceStationID.kBlue1)
                        elif alliance_station_sig.value == "Blue 2":
                            DriverStationSim.setAllianceStationId(AllianceStationID.kBlue2)
                        elif alliance_station_sig.value == "Blue 3":
                            DriverStationSim.setAllianceStationId(AllianceStationID.kBlue3)
                        else:
                            DriverStationSim.setAllianceStationId(AllianceStationID.kUnknown)

                    DriverStationSim.notifyNewData()

        @classmethod
        def get_instance(cls):
            if cls.__instance is None:
                cls.__instance = ReplayAutoEnable()
            return cls.__instance

        def start(self):
            """
            Starts automatically enabling the robot in replay.
            """
            with self.__lock:
                if self.__start_count == 0:
                    # start if we were previously at 0
                    self.__enable_notifier.startPeriodic(0.010)
                self.__start_count += 1

        def stop(self):
            """
            Stops automatically enabling the robot in replay. The
            replay enable will only be stopped when all actuators
            have requested to stop the replay enable.
            """
            with self.__lock:
                if self.__start_count > 0:
                    self.__start_count -= 1
                    if self.__start_count == 0:
                        self.__enable_notifier.stop()

except ImportError:
    pass
