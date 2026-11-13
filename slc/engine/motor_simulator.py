import numpy as np

from slc.dataclass.motor_parameters import MotorParameters
from typing import Tuple, List, Dict, Any

class MotorSimulator:
    def __init__(self, motor_params: MotorParameters, dt: float = 0.001):
        self.motor_params = motor_params
        self.dt = dt
        self.state = np.zeros(2)  # [current, angular_velocity]
        self.delay_steps = int(round(self.motor_params.dead_time / self.dt)) if self.motor_params.dead_time is not None else 0
        self.voltage_buffer: List[float] = [0.0] * max(1, self.delay_steps)
        self.last_voltage: float = 0.0