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

    def step(self, voltage_command: float, tau: float = 0.0) -> Dict[str, float]:
        # Voltage saturation clamping, ensure that the voltage command is within the allowed range
        voltage_command = np.clip(voltage_command, -self.motor_params.max_voltage, self.motor_params.max_voltage)
        # Actuator slew rate limiting by constraining the change in voltage
        max_delta = self.motor_params.slew_rate * self.dt
        voltage_command = np.clip(voltage_command, self.last_voltage - max_delta, self.last_voltage + max_delta)
        self.last_voltage = voltage_command
        self.voltage_buffer.append(voltage_command)
        v_delayed = self.voltage_buffer.pop(0)  # Get the oldest voltage in the buffer

        # Runge-Kutta 4th order integration
        y = self.state
        dt = self.dt
        k1 = self.derivatives(y, v_delayed, tau)
        k2 = self.derivatives(y + 0.5 * dt * k1, v_delayed, tau)
        k3 = self.derivatives(y + 0.5 * dt * k2, v_delayed, tau)
        k4 = self.derivatives(y + dt * k3, v_delayed, tau)
        self.state = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        current, omega = self.state
        rpm = omega * 60 / (2.0 * np.pi)

        return {
            "voltage_command": voltage_command,
            "voltage_delayed": v_delayed,
            "current": current,
            "angular_velocity": omega,
            "rpm": rpm,
            "torque": tau
        }
