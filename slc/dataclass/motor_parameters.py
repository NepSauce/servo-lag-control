from dataclasses import dataclass

@dataclass
class MotorParameters:
    resistance: float | None = None
    inductance: float | None = None
    torque_constant: float | None = None
    back_emf_constant: float | None = None
    inertia: float | None = None
    damping: float | None = None
    voltage_sat: float | None = None
    slew_rate: float | None = None
    dead_time: float | None = None
