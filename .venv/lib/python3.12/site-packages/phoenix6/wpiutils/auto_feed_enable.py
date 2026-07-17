"""
Copyright (C) Cross The Road Electronics.  All rights reserved.
License information can be found in CTRE_LICENSE.txt
For support and suggestions contact support@ctr-electronics.com or file
an issue tracker at https://github.com/CrossTheRoadElec/Phoenix-Releases
"""

try:
    from wpilib import Notifier, DriverStation
    from threading import RLock
    from typing import final
    from phoenix6.signal_logger import SignalLogger
    from phoenix6.unmanaged import feed_enable

    @final
    class AutoFeedEnable:
        __instance = None

        def __init__(self):
            self.__enable_notifier = Notifier(self.__run)
            self.__lock = RLock()
            self.__start_count = 0

        def __run(self):
            if DriverStation.isEnabled():
                feed_enable(0.100)

                if DriverStation.isAutonomous():
                    robot_mode = "Autonomous"
                elif DriverStation.isTest():
                    robot_mode = "Test"
                else:
                    robot_mode = "Teleop"
            else:
                feed_enable(0.0)

                if DriverStation.isEStopped():
                    robot_mode = "EStopped"
                else:
                    robot_mode = "Disabled"

            alliance_station = ""
            alliance = DriverStation.getAlliance()
            location = DriverStation.getLocation()
            if alliance is not None and location is not None:
                if alliance == DriverStation.Alliance.kRed:
                    if location == 1:
                        alliance_station = "Red 1"
                    elif location == 2:
                        alliance_station = "Red 2"
                    elif location == 3:
                        alliance_station = "Red 3"
                elif alliance == DriverStation.Alliance.kBlue:
                    if location == 1:
                        alliance_station = "Blue 1"
                    elif location == 2:
                        alliance_station = "Blue 2"
                    elif location == 3:
                        alliance_station = "Blue 3"

            SignalLogger.write_string("RobotMode", robot_mode)
            SignalLogger.write_string("AllianceStation", alliance_station)
            SignalLogger.write_boolean("DS:IsDSAttached", DriverStation.isDSAttached())
            SignalLogger.write_boolean("DS:IsFMSAttached", DriverStation.isFMSAttached())

        @classmethod
        def get_instance(cls):
            if cls.__instance is None:
                cls.__instance = AutoFeedEnable()
            return cls.__instance

        def start(self):
            """
            Starts feeding the enable signal to CTRE actuators.
            """
            with self.__lock:
                if self.__start_count == 0:
                    # start if we were previously at 0
                    self.__enable_notifier.startPeriodic(0.010)
                self.__start_count += 1

        def stop(self):
            """
            Stops feeding the enable signal to CTRE actuators. The
            enable signal will only be stopped when all actuators
            have requested to stop the enable signal.
            """
            with self.__lock:
                if self.__start_count > 0:
                    self.__start_count -= 1
                    if self.__start_count == 0:
                        self.__enable_notifier.stop()

except ImportError:
    pass
