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

    def reset(self, initial_state: Tuple[float, float] = (0.0, 0.0)):
        self.state = np.array(initial_state, dtype=float)
        self.voltage_buffer = [0.0] * len(self.voltage_buffer)
        self.last_voltage = 0.0

    def derivatives(self, state: np.ndarray, voltage: float, tau: float):
        i, omega = state
        di_dt = (voltage - self.motor_params.resistance * i - self.motor_params.back_emf_constant * omega) / self.motor_params.inductance
        domega_dt = (self.motor_params.torque_constant * i - self.motor_params.damping * omega - tau) / self.motor_params.inertia
        
        return np.array([di_dt, domega_dt])