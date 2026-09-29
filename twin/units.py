"""Unit conversion to SI. Unknown units raise, so nothing is silently mis-scaled.

Temperatures are kept in degrees Celsius throughout the code base (unit 'degC');
functions that need kelvin convert explicitly with `to_kelvin`.
"""

G = 9.80665  # standard gravity, m/s2 (CGPM 1901 definition)
BBL = 0.158987294928  # m3 per US oil barrel (exact: 42 US gal)
KELVIN_OFFSET = 273.15

_TO_SI = {
    "dimensionless": 1.0,
    "fraction": 1.0,
    "m": 1.0,
    "mD": 9.869233e-16,  # 1 darcy = 9.869233e-13 m2
    "Pa": 1.0,
    "kPa": 1e3,
    "MPa": 1e6,
    "kPa/m": 1e3,
    "cP": 1e-3,
    "1/Pa": 1.0,
    "1/K": 1.0,
    "K": 1.0,  # used only for temperature differences / std-devs
    "J/m3.K": 1.0,
    "J/kg.K": 1.0,
    "W/m.K": 1.0,
    "W/m2.K": 1.0,
    "m2/s": 1.0,
    "kg/m3": 1.0,
    "N": 1.0,
    "kg": 1.0,
    "t": 1000.0,  # tonne -> kg
    "t/d": 1000.0 / 86400.0,  # -> kg/s
    "d": 86400.0,  # -> s
    "m3/d": 1.0 / 86400.0,  # -> m3/s
    # currency units are passed through unchanged
    "USD/bbl": 1.0,
    "USD/GJ": 1.0,
    "USD/kWh": 1.0,
    "USD/d": 1.0,
}
_PASSTHROUGH = {"degC", "date"}


def to_si(value, unit):
    if unit in _PASSTHROUGH:
        return value
    if unit not in _TO_SI:
        raise ValueError(f"Unknown unit '{unit}' - add it to twin/units.py explicitly")
    f = _TO_SI[unit]
    if isinstance(value, (list, tuple)):
        return [v * f for v in value]
    return value * f


def to_kelvin(t_c):
    return t_c + KELVIN_OFFSET
