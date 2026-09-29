"""Physics-based CSS + SRP well simulator (the "twin" engine), daily time step.

Cycle = injection -> soak -> production (until cut-off). Well-to-surface coupling:

  reservoir: two pressure regions
     inner region (radius r_in): heated zone (r_h, Marx-Langenheim + lumped cooling)
                                 + condensate bank beyond it (water saturation S_wc + dS_bank)
     outer region (r_in .. r_e): cold oil at reservoir temperature
     well <- inner : 2 pi k h (p_in - p_wf) / R_in
     inner <- outer: 2 pi k h (p_e - p_in) / R_out
     R_in  = (ln(r_1/r_w) + skin)/lambda_1 + ln(r_in/r_1)/lambda_bank
     R_out = ln(r_e/r_in)/lambda_cold
     each region stores fluid through an effective compressibility; the two pressures are
     advanced with a backward-Euler (implicit) 2x2 solve every day.
  + gravity drainage of heated oil inside the heated zone (Dupuit unconfined radial flow,
     Dupuit 1863):  q_g = pi k kro_e (rho_o - rho_v) g H^2 / (mu_o(T_h) (ln(r_h/r_w) + skin)),
     H = h * (drainable heated oil remaining / drainable pore volume); vapour density neglected
  -> tubing temperature profile (Ramey) -> viscosity along tubing
  -> SRP analytic model (capacity, floating, Goodman, energy)
  -> production = min(potential at pumped-off intake, pump capacity); if pump-limited the
     region pressures are re-solved with the well rate fixed to the pump capacity.

Main simplifications (documented in README):
  * single layer, radially symmetric; no steam flashing / solution-gas drive;
  * effective (tank) compressibility; injection over-pressure in the outer region leaks
    off with a time constant, and a floor on outer pressure represents unmodelled support;
  * surface injection pressure = bottom-hole injection pressure;
  * tubing temperature profile uses the previous day's liquid rate (1-day lag).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..units import BBL, G
from . import steam as stm
from .fluids import OilPVT
from .srp_analytic import SRPDesign, kinematics, motor_power, operating_point
from .thermal import conduction_conductances, marx_langenheim_area
from .wellbore import (production_temperature_profile, resistance_per_length,
                       steam_heat_loss_rate)

DAY = 86400.0


@dataclass
class Well:
    well_id: str
    k: float
    h: float
    phi: float
    s_oi: float
    r_e: float
    skin: float
    p_i: float
    t_r: float
    api: float
    mu_ref: float
    walther_b: float
    stroke: float


@dataclass
class CycleControls:
    steam_mass: float  # kg
    inj_pressure: float  # Pa (bottom-hole = surface assumed)
    quality_surface: float
    soak_days: int
    cutoff_rate: float  # m3/s stock-tank oil
    max_prod_days: int
    spm_policy: str  # "manual" | "twin"
    spm_init: float = 3.0
    ratio_init: float = 1.0
    min_prod_days: int | None = None  # cut-off test not applied before this day
    lift_cutoff: bool = True  # twin only: end the cycle when lifting is infeasible (see config)


@dataclass
class WellState:
    n_rem: float  # oil in place in drainage volume, reservoir m3
    w_in: float  # injected-water (condensate) mass in inner region, kg
    w_out: float  # injected-water mass that has moved to the outer region, kg
    e_h: float  # heated-zone energy above reservoir temperature, J
    r_h: float  # heated radius, m
    p_in: float  # inner-region pressure, Pa
    p_e: float  # outer-region pressure, Pa
    p_cap: float  # cap on pressures (last injection pressure)
    t_heat: float  # s since start of last injection
    cum_oil: float = 0.0  # stock-tank m3
    cum_water: float = 0.0
    cum_steam: float = 0.0  # kg
    cycle_no: int = 0
    last_q_liq: float = 0.0  # m3/s
    day: int = 0  # days since the well record starts
    leaked_water: float = 0.0  # kg lost beyond the drainage area
    n_h: float = 0.0  # drainable (above S_or) oil in the heated zone, reservoir m3
    r_max: float = 0.0  # largest radius ever heated (rock beyond it is uncontacted)
    heat_in: float = 0.0  # J stored by injections (after Marx-Langenheim losses)
    heat_cond_loss: float = 0.0  # J lost to cap/base rock after injection
    heat_produced: float = 0.0  # J carried out with produced fluids
    water_prod_mass: float = 0.0  # kg of injected water produced back
    water_injected: float = 0.0  # kg


class CSSWellSimulator:
    def __init__(self, cfg, well: Well, settings: dict, water: stm.WaterProps,
                 rng: np.random.Generator):
        self.cfg, self.w, self.s, self.water, self.rng = cfg, well, settings, water, rng
        c = cfg
        self.oil = OilPVT.from_config(c, well.api, well.mu_ref, well.walther_b)
        self.depth = c.value("reservoir.depth")
        self.srp = SRPDesign.from_config(c, self.depth, well.stroke)
        self.t_surf = c.value("wellbore.surface_temperature")
        self.grad = (well.t_r - self.t_surf) / self.depth  # derived geothermal gradient
        self.r_w = c.value("reservoir.wellbore_radius")
        self.s_wc = c.value("reservoir.connate_water_saturation")
        self.s_or = c.value("reservoir.residual_oil_saturation")
        self.n_o = c.value("reservoir.corey_exponent_oil")
        self.n_w = c.value("reservoir.corey_exponent_water")
        self.krw_e = c.value("reservoir.krw_endpoint")
        self.kro_e = c.value("reservoir.kro_endpoint")
        self.c_eff = c.value("reservoir.effective_compressibility")
        self.ds_bank = c.value("reservoir.condensate_bank_saturation_increase")
        self.tau_diss = c.value("reservoir.pressure_dissipation_time")
        self.tau_disp = c.value("reservoir.condensate_dispersal_time")
        self.p_min = c.value("reservoir.min_drainage_pressure_fraction") * well.p_i
        self.p_frac = c.value("reservoir.fracture_gradient") * self.depth
        self.m_r = c.value("reservoir.rock_volumetric_heat_capacity")
        self.lam_r = c.value("reservoir.rock_thermal_conductivity")
        self.alpha_r = c.value("reservoir.rock_thermal_diffusivity")
        self.lam_c = c.value("reservoir.caprock_thermal_conductivity")
        self.alpha_c = c.value("reservoir.caprock_thermal_diffusivity")
        self.b_o = c.value("oil.formation_volume_factor")
        self.wc_inv = c.value("oil.emulsion_inversion_watercut")
        self.u_vit = c.value("wellbore.vit_overall_U")
        self.r_to = c.value("wellbore.tubing_outer_radius")
        self.k_e = c.value("wellbore.earth_thermal_conductivity")
        self.alpha_e = c.value("wellbore.earth_thermal_diffusivity")
        self.p_wh = c.value("wellbore.wellhead_pressure")
        self.p_cas = c.value("wellbore.casing_pressure")
        self.h_sub = c.value("wellbore.min_pump_submergence")
        self.gen_rate = c.value("steam.generator_max_rate")
        self.boiler_eff = c.value("steam.boiler_efficiency")
        self.eta_surf = c.value("srp.surface_efficiency")
        self.spm_min = c.value("srp.spm_min")
        self.spm_max = c.value("srp.spm_max")
        self.ratio_max = c.value("srp.max_speed_ratio")
        self.react = int(c.value("srp.manual_reaction_days"))
        self.spm_step = c.value("srp.manual_spm_step")
        self.low_fill = c.value("srp.low_fillage_threshold")
        self.target_fill = c.value("srp.target_fillage")
        self.float_margin = c.value("srp.floating_safety_margin")
        self.infeasible_cut = int(c.value("srp.twin_infeasible_days_to_cutoff"))
        self.noise_rate = c.value("measurement_noise.rate_relative_sd")
        self.noise_temp = c.value("measurement_noise.temperature_sd")
        self.v_p = math.pi * well.r_e**2 * well.h * well.phi  # drainage pore volume
        self.rho_w_res = float(water.density(well.t_r))
        self.mu_cold = float(self.oil.viscosity(well.t_r))
        self.mu_w_res = float(water.viscosity(well.t_r))
        n = int(settings["tubing_nodes"])
        self.dz = self.srp.pump_depth / n
        self.y_nodes = (np.arange(n) + 0.5) * self.dz  # height above pump
        self.spm_grid = np.arange(self.spm_min, self.spm_max + 1e-9, settings["spm_grid_step"])
        self.ratio_grid = np.arange(1.0, self.ratio_max + 1e-9, settings["ratio_grid_step"])

    # ------------------------------------------------------------------ state
    def initial_state(self) -> WellState:
        w = self.w
        return WellState(n_rem=self.v_p * w.s_oi, w_in=0.0, w_out=0.0, e_h=0.0, r_h=0.0,
                         p_in=w.p_i, p_e=w.p_i, p_cap=w.p_i, t_heat=0.0)

    @property
    def p_wf_min(self):
        """Pumped-off intake pressure: casing pressure + minimum submergence of liquid."""
        return self.p_cas + float(self.oil.density(self.w.t_r)) * G * self.h_sub

    def heated_temperature(self, st: WellState):
        if st.r_h <= self.r_w or st.e_h <= 0:
            return self.w.t_r
        return self.w.t_r + st.e_h / (self.m_r * self.w.h * math.pi * st.r_h**2)

    def heat_step(self, st: WellState, dt: float, u_prod: float = 0.0):
        """Advance heated-zone energy by dt.

        Losses leaving the reservoir (cap/base conduction + produced fluids, u_prod W/K) are
        integrated exponentially. Lateral conduction heat stays in the reservoir: it heats an
        adjacent ring to the zone temperature, so the zone grows (energy conserved) and the
        ring's mobile oil joins the drainable heated-oil inventory n_h.
        Returns (heat lost to cap/base J, heat carried by produced fluids J)."""
        w = self.w
        if st.e_h <= 0 or st.r_h <= self.r_w:
            return 0.0, 0.0
        u_tb, u_side = conduction_conductances(st.r_h, w.h, st.t_heat + 0.5 * dt, self.lam_c,
                                               self.alpha_c, self.lam_r, self.alpha_r)
        area = math.pi * st.r_h**2
        cap_heat = self.m_r * w.h * area  # J/K
        d_t = st.e_h / cap_heat
        e0 = st.e_h
        st.e_h *= math.exp(-(u_tb + u_prod) * dt / cap_heat)
        lost = e0 - st.e_h
        frac_tb = u_tb / (u_tb + u_prod) if (u_tb + u_prod) > 0 else 0.0
        # lateral: heat q_side moves into a ring that is brought to the zone temperature
        q_side = u_side * d_t * dt
        d_area = q_side / (self.m_r * w.h * d_t)
        new_area = min(area + d_area, math.pi * (0.95 * w.r_e) ** 2)
        st.r_h = math.sqrt(new_area / math.pi)
        self._contact(st)
        return lost * frac_tb, lost * (1 - frac_tb)

    def _contact(self, st: WellState):
        """Rock heated for the first time brings its mobile oil into the drainable inventory;
        the inventory is capped by the drainable pore volume of the current hot zone."""
        w = self.w
        if st.r_h > st.r_max:
            s_o_avg = st.n_rem / self.v_p
            ring = math.pi * (st.r_h**2 - st.r_max**2) * w.h * w.phi
            st.n_h += ring * max(s_o_avg - self.s_or, 0.0)
            st.r_max = st.r_h
        cap = w.phi * math.pi * st.r_h**2 * w.h * (1.0 - self.s_wc - self.s_or)
        st.n_h = min(st.n_h, cap)

    def gravity_drainage(self, st: WellState, t_h: float):
        """Dupuit gravity-drainage rate (m3/s) of heated oil toward a pumped-off well."""
        w = self.w
        if st.r_h <= self.r_w or st.e_h <= 0 or st.n_h <= 0:
            return 0.0
        vp_h = w.phi * math.pi * st.r_h**2 * w.h
        col = w.h * min(st.n_h / (vp_h * (1.0 - self.s_wc - self.s_or)), 1.0)
        mu_o = float(self.oil.viscosity(t_h))
        res = math.log(st.r_h / self.r_w) + w.skin
        return (math.pi * w.k * self.kro_e * float(self.oil.density(t_h)) * G * col**2
                / (mu_o * max(res, 1e-6)))

    def _corey(self, s_w, f_dep):
        s_wn = min(max((s_w - self.s_wc) / (1.0 - self.s_wc - self.s_or), 0.0), 1.0)
        return self.kro_e * (1.0 - s_wn) ** self.n_o * f_dep, self.krw_e * s_wn**self.n_w

    # --------------------------------------------------------------- regions
    def regions(self, st: WellState, t_h: float):
        w = self.w
        s_o_avg = st.n_rem / self.v_p
        f_dep = min(max((s_o_avg - self.s_or) / (w.s_oi - self.s_or), 0.0), 1.0)
        lam_c = self.kro_e * f_dep / self.mu_cold
        heated = st.r_h > self.r_w and st.e_h > 0
        r1 = st.r_h if heated else self.r_w
        rho_w1 = float(self.water.density(t_h))
        w_vol = st.w_in / rho_w1
        vp1 = w.phi * math.pi * r1**2 * w.h
        ds1 = min(w_vol / vp1, self.ds_bank) if (heated and vp1 > 0) else 0.0
        w_excess = w_vol - ds1 * vp1
        r_in = math.sqrt(r1**2 + max(w_excess, 0.0) / (math.pi * w.h * w.phi * self.ds_bank))
        r_in = min(r_in, 0.95 * w.r_e)
        # bank (at reservoir temperature)
        kro_b, krw_b = self._corey(self.s_wc + self.ds_bank, f_dep)
        lam_b = kro_b / self.mu_cold + krw_b / self.mu_w_res
        fw_b = (krw_b / self.mu_w_res) / lam_b if lam_b > 0 else 0.0
        if heated:
            kro, krw = self._corey(self.s_wc + ds1, f_dep)
            mu_o, mu_w = float(self.oil.viscosity(t_h)), float(self.water.viscosity(t_h))
            lam_1 = kro / mu_o + krw / mu_w
            fw = (krw / mu_w) / lam_1 if lam_1 > 0 else 0.0
        elif w_vol > 0:
            lam_1, fw = lam_b, fw_b
        else:
            lam_1, fw = lam_c, 0.0
        if lam_1 <= 0:
            r_in_res = math.inf
        else:
            r_in_res = (math.log(r1 / self.r_w) + w.skin) / lam_1
            if r_in > r1:
                r_in_res += math.log(r_in / r1) / lam_b
            r_in_res = max(r_in_res, 1e-12)
        r_out_res = math.log(w.r_e / r_in) / lam_c if lam_c > 0 else math.inf
        return dict(r1=r1, r_in=r_in, R_in=r_in_res, R_out=r_out_res, fw=fw,
                    v_in=w.phi * math.pi * r_in**2 * w.h,
                    v_out=w.phi * math.pi * (w.r_e**2 - r_in**2) * w.h,
                    w_vol=w_vol, rho_w1=rho_w1, f_dep=f_dep)

    def _solve_pressures(self, st, g, dt, p_wf=None, q_fixed=None):
        """Backward-Euler step of the two-region pressure system.
        Either the bottom-hole pressure p_wf or the well rate q_fixed is prescribed."""
        tr = 2.0 * math.pi * self.w.k * self.w.h
        j_i = tr / g["R_in"] if math.isfinite(g["R_in"]) else 0.0
        j_o = tr / g["R_out"] if math.isfinite(g["R_out"]) else 0.0
        c_in = self.c_eff * g["v_in"] / dt
        c_out = self.c_eff * g["v_out"] / dt
        if q_fixed is None:
            a = np.array([[c_in + j_o + j_i, -j_o], [-j_o, c_out + j_o]])
            b = np.array([c_in * st.p_in + j_i * p_wf, c_out * st.p_e])
        else:
            a = np.array([[c_in + j_o, -j_o], [-j_o, c_out + j_o]])
            b = np.array([c_in * st.p_in - q_fixed, c_out * st.p_e])
        p_in, p_e = np.linalg.solve(a, b)
        q_well = q_fixed if q_fixed is not None else max(j_i * (p_in - p_wf), 0.0)
        return float(p_in), float(p_e), float(q_well), float(j_o * (p_e - p_in))

    def _move_water(self, st, q_oi, dt, rho_w1):
        """Bookkeeping of injected water crossing the inner/outer boundary."""
        vol = abs(q_oi) * dt
        if q_oi > 0:  # outer -> inner: injected water that went outward returns first
            m = min(vol * self.rho_w_res, st.w_out)
            st.w_out -= m
            st.w_in += m
        elif q_oi < 0:  # inner -> outer: water leaves the bank
            m = min(vol * rho_w1, st.w_in)
            st.w_in -= m
            st.w_out += m

    def _disperse(self, st, g, dt):
        """Condensate in the bank beyond the heated zone disperses into the formation."""
        vp1 = self.w.phi * math.pi * g["r1"] ** 2 * self.w.h
        held = min(self.ds_bank * vp1 * g["rho_w1"], st.w_in) if g["r1"] > self.r_w else 0.0
        excess = st.w_in - held
        if excess > 0:
            m = excess * (1.0 - math.exp(-dt / self.tau_disp))
            st.w_in -= m
            st.leaked_water += m

    def _leak_off(self, st, g, dt):
        """Outer over-pressure above initial pressure decays with time constant tau; the
        leaked volume is injected water leaving the drainage area. Below p_min the outer
        pressure is held (unmodelled support)."""
        if st.p_e > self.w.p_i:
            new = self.w.p_i + (st.p_e - self.w.p_i) * math.exp(-dt / self.tau_diss)
            m = min(self.c_eff * g["v_out"] * (st.p_e - new) * self.rho_w_res, st.w_out)
            st.w_out -= m
            st.leaked_water += m
            st.p_e = new
        st.p_e = max(st.p_e, self.p_min)

    # ------------------------------------------------------------------ SRP
    def srp_conditions(self, t_bottom, q_liq_prev, wc, days_on, spm, ratio):
        rho_o = float(self.oil.density(t_bottom))
        rho_w = float(self.water.density(t_bottom))
        rho_mix = (1 - wc) * rho_o + wc * rho_w
        cp_mix_vol = (1 - wc) * rho_o * self.oil.cp + wc * rho_w * float(self.water.cp(t_bottom))
        r_prime = resistance_per_length(max(days_on, 1) * DAY, self.r_to, self.u_vit, self.k_e,
                                        self.alpha_e, self.r_w)
        t_prof = production_temperature_profile(t_bottom, self.y_nodes, self.srp.pump_depth,
                                                self.t_surf, self.grad, q_liq_prev * cp_mix_vol,
                                                r_prime)
        if wc >= self.wc_inv:
            mu_prof, mu_pump = self.water.viscosity(t_prof), float(self.water.viscosity(t_bottom))
        else:
            mu_prof, mu_pump = self.oil.viscosity(t_prof), float(self.oil.viscosity(t_bottom))
        op = operating_point(self.srp, mu_prof, self.dz, mu_pump, rho_mix, self.p_wf_min,
                             self.p_wh, spm, ratio, self.eta_surf)
        op["t_wellhead"] = float(t_prof[-1])
        op["mu_pump"] = mu_pump
        op["stroke_fraction"] = self._stroke_fraction(op, spm, ratio)
        return op

    def tubing_viscosity_nodes(self, t_bottom, q_liq_prev, wc, days_on, n_nodes):
        """Viscosity at n_nodes rod nodes from surface (index 0) to pump, and mixture density -
        the same thermal/mixture model the daily SRP evaluation uses (for FD verification)."""
        rho_o, rho_w = float(self.oil.density(t_bottom)), float(self.water.density(t_bottom))
        rho_mix = (1 - wc) * rho_o + wc * rho_w
        cp_mix_vol = (1 - wc) * rho_o * self.oil.cp + wc * rho_w * float(self.water.cp(t_bottom))
        r_prime = resistance_per_length(max(days_on, 1) * DAY, self.r_to, self.u_vit, self.k_e,
                                        self.alpha_e, self.r_w)
        y = np.linspace(self.srp.pump_depth, 0.0, n_nodes)  # height above pump, surface first
        t_prof = production_temperature_profile(t_bottom, y, self.srp.pump_depth, self.t_surf,
                                                self.grad, q_liq_prev * cp_mix_vol, r_prime)
        mu = self.water.viscosity(t_prof) if wc >= self.wc_inv else self.oil.viscosity(t_prof)
        return np.asarray(mu, float), rho_mix

    def _stroke_fraction(self, op, spm, ratio):
        """Lost stroke when floating: rods fall at terminal velocity (rigid-rod estimate)."""
        k = kinematics(self.srp.stroke, spm, ratio)
        c_tot = op["c_rod"] + op["c_plunger"]
        v_term = max(op["buoyant_weight"] - self.srp.f_coulomb, 0.0) / c_tot
        return np.minimum(1.0, v_term * k["t_dn"] / self.srp.stroke)

    def choose_twin(self, t_bottom, q_prev, wc, days_on, q_pot):
        """Twin controller: pick (SPM, speed ratio) on a grid.
        Feasible = no floating (with margin) and Goodman loading <= 1.
        Among feasible: if capacity*target_fillage can match the inflow potential, use the
        least-power point that does; otherwise maximise rate (ties -> least power).
        No feasible point: least constraint violation, then least power."""
        spm = self.spm_grid[:, None]
        ratio = self.ratio_grid[None, :]
        op = self.srp_conditions(t_bottom, q_prev, wc, days_on, spm, ratio)
        cap = op["capacity"] * op["stroke_fraction"]
        feasible = (op["prl_min"] >= self.float_margin * op["buoyant_weight"]) & (op["goodman"] <= 1.0)
        q = np.minimum(q_pot, cap)
        power = (q * max(op["p_discharge"] - self.p_wf_min, 0.0) + op["friction_power"]) / self.eta_surf
        power = np.broadcast_to(power, cap.shape)
        if feasible.any():
            meets = feasible & (cap * self.target_fill >= q_pot)
            if meets.any():
                idx = np.unravel_index(np.argmin(np.where(meets, power, np.inf)), cap.shape)
            else:
                qf = np.where(feasible, q, -1.0)
                cand = feasible & (qf >= qf.max() * (1 - 1e-9))
                idx = np.unravel_index(np.argmin(np.where(cand, power, np.inf)), cap.shape)
            ok = True
        else:
            wb = op["buoyant_weight"]
            viol = (np.maximum(self.float_margin * wb - op["prl_min"], 0.0) / wb
                    + np.maximum(op["goodman"] - 1.0, 0.0))
            viol = np.broadcast_to(viol, cap.shape)
            cand = viol <= viol.min() * (1 + 1e-6) + 1e-12
            idx = np.unravel_index(np.argmin(np.where(cand, power, np.inf)), cap.shape)
            ok = False
        return float(self.spm_grid[idx[0]]), float(self.ratio_grid[idx[1]]), ok

    # ------------------------------------------------------------ production
    def _noisy(self, x, sd):
        return max(0.0, x * (1.0 + sd * self.rng.standard_normal()))

    def produce(self, st: WellState, controls: CycleControls, phase="production", records=None,
                min_days=None, window=None):
        records = [] if records is None else records
        for _ in self.produce_iter(st, controls, phase, records, min_days, window):
            pass
        return records

    def produce_iter(self, st: WellState, controls: CycleControls, phase="production", records=None,
                     min_days=None, window=None):
        """Production period as a generator: yields each day's record (used by the live twin).
        `controls.spm_policy` / `lift_cutoff` are re-read every day, so they can be switched live."""
        s = self.s
        if min_days is None:
            min_days = (controls.min_prod_days if controls.min_prod_days is not None
                        else s["min_days_before_cutoff"])
        window = s["cutoff_window_days"] if window is None else window
        no_resp = int(s["no_response_days"])
        records = [] if records is None else records
        spm, ratio = controls.spm_init, controls.ratio_init
        cnt = {"float": 0, "low": 0, "high": 0}
        oil_hist, peak_avg = [], 0.0
        infeasible_run = 0
        p_wf = self.p_wf_min
        for d in range(1, controls.max_prod_days + 1):
            t_h = self.heated_temperature(st)
            g = self.regions(st, t_h)
            p_in_pot, p_e_pot, q_press_pot, q_oi_pot = self._solve_pressures(st, g, DAY, p_wf=p_wf)
            q_gd_pot = min(self.gravity_drainage(st, t_h), st.n_h / DAY)
            q_pot = q_press_pot + q_gd_pot
            # expected water cut of the produced stream (gravity-drained oil is water-free),
            # used for the tubing mixture in the SRP evaluation
            wc = (min(g["fw"] * q_press_pot, g["w_vol"] / DAY) / q_pot) if q_pot > 0 else g["fw"]
            wc_srp = wc
            q_prev = st.last_q_liq if st.last_q_liq > 0 else q_pot
            twin_ok = True
            if controls.spm_policy == "twin":
                spm, ratio, twin_ok = self.choose_twin(t_h, q_prev, wc, d, q_pot)
            elif controls.spm_policy != "manual":
                raise ValueError(controls.spm_policy)
            op = self.srp_conditions(t_h, q_prev, wc, d, spm, ratio)
            cap = float(op["capacity"] * op["stroke_fraction"])
            if q_pot > cap:  # pump-limited: gravity-drained oil first, pressure flow gets the rest
                q_gd = min(q_gd_pot, cap)
                p_in, p_e, q_press, q_oi = self._solve_pressures(st, g, DAY, q_fixed=cap - q_gd)
            else:
                q_gd = q_gd_pot
                p_in, p_e, q_press, q_oi = p_in_pot, p_e_pot, q_press_pot, q_oi_pot
            q_liq = q_press + q_gd
            fill = q_liq / cap if cap > 0 else 0.0
            floating = bool(op["prl_min"] < 0)
            w_avail = g["w_vol"] / DAY
            q_w = min(g["fw"] * q_press, w_avail)
            q_o = q_liq - q_w
            wc = q_w / q_liq if q_liq > 0 else wc
            power = motor_power(op, q_liq, p_wf)
            # --- heat balance
            u_p = (float(self.oil.density(t_h)) * self.oil.cp * q_o
                   + g["rho_w1"] * float(self.water.cp(t_h)) * q_w)
            h_cond, h_prod = self.heat_step(st, DAY, u_p)
            st.heat_cond_loss += h_cond
            st.heat_produced += h_prod
            # --- mass balance
            m_w = min(g["rho_w1"] * q_w * DAY, st.w_in)
            st.w_in -= m_w
            st.water_prod_mass += m_w
            self._move_water(st, q_oi, DAY, g["rho_w1"])
            st.n_rem -= q_o * DAY
            st.n_h = max(st.n_h - q_gd * DAY, 0.0)
            st.p_in, st.p_e = min(p_in, st.p_cap), min(p_e, st.p_cap)
            self._leak_off(st, g, DAY)
            self._disperse(st, g, DAY)
            st.t_heat += DAY
            oil_st = q_o / self.b_o * DAY  # stock-tank m3/d
            st.cum_oil += oil_st
            st.cum_water += q_w * DAY
            st.last_q_liq = q_liq
            st.day += 1
            rec = dict(day=st.day, phase=phase, cycle_no=st.cycle_no, prod_day=d,
                       t_bottom_true=t_h, oil_m3d_true=oil_st, water_m3d_true=q_w * DAY,
                       oil_m3d=self._noisy(oil_st, self.noise_rate),
                       water_m3d=self._noisy(q_w * DAY, self.noise_rate),
                       t_bottom=t_h + self.noise_temp * self.rng.standard_normal(),
                       t_wellhead=op["t_wellhead"] + self.noise_temp * self.rng.standard_normal(),
                       spm=spm, speed_ratio=ratio, fillage=fill, floating=floating,
                       twin_feasible=twin_ok,
                       impact_strokes=spm * 1440.0 if floating else 0.0,
                       prl_min_kN=float(op["prl_min"]) / 1e3, prl_max_kN=float(op["prl_max"]) / 1e3,
                       goodman=float(op["goodman"]), motor_kwh=power * DAY / 3.6e6,
                       potential_m3d=q_pot * DAY, capacity_m3d=cap * DAY,
                       gravity_m3d=q_gd * DAY, water_cut=wc,
                       q_prev_m3d=q_prev * DAY, wc_srp=wc_srp,
                       p_in_MPa=st.p_in / 1e6, p_e_MPa=st.p_e / 1e6,
                       mu_pump_cP=op["mu_pump"] * 1e3, r_inner=g["r_in"])
            records.append(rec)
            oil_hist.append(rec["oil_m3d"])  # operator sees the measured rate
            if controls.spm_policy == "manual":  # reactive operator rules
                cnt["float"] = cnt["float"] + 1 if floating else 0
                cnt["low"] = cnt["low"] + 1 if fill < self.low_fill else 0
                cnt["high"] = cnt["high"] + 1 if (q_pot >= cap and not floating) else 0
                if cnt["float"] >= self.react or cnt["low"] >= self.react:
                    spm = max(self.spm_min, spm - self.spm_step)
                    cnt = dict.fromkeys(cnt, 0)
                elif cnt["high"] >= self.react:
                    spm = min(self.spm_max, spm + self.spm_step)
                    cnt = dict.fromkeys(cnt, 0)
            # cut-off: measured oil rate (moving average) fell below the cut-off after having
            # exceeded it, or never exceeded it within `no_response_days`
            avg = float(np.mean(oil_hist[-window:]))
            if d >= window:
                peak_avg = max(peak_avg, avg)
            cut = controls.cutoff_rate * DAY
            stop = None
            if d >= min_days and avg < cut and (peak_avg >= cut or d >= no_resp):
                stop = "rate_cutoff"
            elif controls.spm_policy == "twin" and controls.lift_cutoff:  # lift-limited cut-off
                infeasible_run = 0 if twin_ok else infeasible_run + 1
                if infeasible_run >= self.infeasible_cut:
                    stop = "lift_cutoff"
            rec["stop_reason"] = stop
            yield rec
            if stop:
                return

    # ----------------------------------------------------------------- cycle
    def _combine_heat(self, e_prev, r_prev, a_ml, d_t):
        """New hot zone = Marx-Langenheim zone of this injection plus the residual energy of
        earlier cycles (energy conserved). The residual energy raises the zone temperature up
        to the steam temperature; beyond that it enlarges the zone."""
        w = self.w
        e_tot = self.m_r * w.h * a_ml * d_t + e_prev
        a = max(a_ml, 1e-9)
        if e_tot / (self.m_r * w.h * a) > d_t:
            a = e_tot / (self.m_r * w.h * d_t)
        a = min(a, math.pi * (0.95 * w.r_e) ** 2)
        e = min(e_tot, self.m_r * w.h * a * d_t)
        return e, math.sqrt(a / math.pi)

    def inject_and_soak(self, st: WellState, c: CycleControls, records: list):
        w = self.w
        if not (max(st.p_e, st.p_in) < c.inj_pressure < self.p_frac):
            raise ValueError(f"injection pressure {c.inj_pressure/1e6:.2f} MPa outside "
                             f"({max(st.p_e, st.p_in)/1e6:.2f}, fracture {self.p_frac/1e6:.2f}) MPa")
        t_s = stm.saturation_temperature(c.inj_pressure)
        h_f, h_fg = stm.h_liquid_sat(c.inj_pressure), stm.latent_heat(c.inj_pressure)
        m_dot = self.gen_rate
        t_inj = c.steam_mass / m_dot
        r_prime = resistance_per_length(0.5 * t_inj, self.r_to, self.u_vit, self.k_e, self.alpha_e,
                                        self.r_w)
        q_loss = steam_heat_loss_rate(t_s, self.depth, self.t_surf, self.grad, r_prime)
        x_b = max(c.quality_surface - q_loss / (m_dot * h_fg), 0.0)
        h_res = stm.h_liquid(w.t_r, c.inj_pressure)
        q_heat = m_dot * (h_f + x_b * h_fg - h_res)
        d_t = t_s - w.t_r
        e_prev, r_prev = st.e_h, st.r_h
        n_h_prev = st.n_h
        st.cycle_no += 1
        st.p_cap = c.inj_pressure
        inj_days = max(1, math.ceil(t_inj / DAY))
        t_done = 0.0
        for _ in range(inj_days):
            dt_inj = min(DAY, t_inj - t_done)
            t_done += dt_inj
            a_ml = marx_langenheim_area(q_heat, t_done, w.h, self.m_r, self.lam_c, self.alpha_c, d_t)
            st.e_h, st.r_h = self._combine_heat(e_prev, r_prev, a_ml, d_t)
            st.w_in += m_dot * dt_inj
            st.water_injected += m_dot * dt_inj
            # inner region held at injection pressure; implicit outer-region update
            g = self.regions(st, t_s)
            tr = 2.0 * math.pi * w.k * w.h
            j_o = tr / g["R_out"] if math.isfinite(g["R_out"]) else 0.0
            c_out = self.c_eff * g["v_out"] / DAY
            p_e_new = (c_out * st.p_e + j_o * c.inj_pressure) / (c_out + j_o)
            q_out = j_o * (c.inj_pressure - p_e_new)
            self._move_water(st, -q_out, DAY, g["rho_w1"])
            st.p_in, st.p_e = c.inj_pressure, p_e_new
            self._leak_off(st, g, DAY)
            self._disperse(st, g, DAY)
            st.day += 1
            records.append(dict(day=st.day, phase="injection", cycle_no=st.cycle_no, prod_day=0,
                                steam_t=m_dot * dt_inj / 1e3, t_bottom_true=t_s,
                                p_in_MPa=st.p_in / 1e6, p_e_MPa=st.p_e / 1e6))
        st.cum_steam += c.steam_mass
        st.heat_in += st.e_h - e_prev
        st.t_heat = t_inj
        r_heated = st.r_h
        self._contact(st)
        # soak: well shut in, conduction losses only, pressures equilibrate
        for _ in range(int(c.soak_days)):
            h_cond, _ = self.heat_step(st, DAY)
            st.heat_cond_loss += h_cond
            g = self.regions(st, self.heated_temperature(st))
            p_in, p_e, _, q_oi = self._solve_pressures(st, g, DAY, q_fixed=0.0)
            self._move_water(st, q_oi, DAY, g["rho_w1"])
            st.p_in, st.p_e = min(p_in, st.p_cap), min(p_e, st.p_cap)
            self._leak_off(st, g, DAY)
            self._disperse(st, g, DAY)
            st.t_heat += DAY
            st.day += 1
            records.append(dict(day=st.day, phase="soak", cycle_no=st.cycle_no, prod_day=0,
                                t_bottom_true=self.heated_temperature(st),
                                p_in_MPa=st.p_in / 1e6, p_e_MPa=st.p_e / 1e6))
        h_feed = stm.h_liquid(self.t_surf, c.inj_pressure)
        fuel_j = c.steam_mass * (h_f + c.quality_surface * h_fg - h_feed) / self.boiler_eff
        return dict(t_steam=t_s, x_bottom=x_b, q_heat_MW=q_heat / 1e6, heat_injected_J=q_heat * t_inj,
                    r_heated=r_heated, inj_days=inj_days, fuel_GJ=fuel_j / 1e9,
                    wellbore_loss_kW=q_loss / 1e3)

    def run_cycle(self, st: WellState, c: CycleControls):
        """Returns (daily records, cycle summary). Mutates st."""
        records = []
        ctx = self.start_cycle(st, c, records)
        self.produce(st, c, records=records)
        return records, self.summarise_cycle(st, c, records, ctx)

    def start_cycle(self, st: WellState, c: CycleControls, records: list):
        """Injection + soak; returns the context summarise_cycle needs."""
        ctx = dict(oil0=st.cum_oil, water0=st.cum_water, start_day=st.day + 1)
        ctx["inj"] = self.inject_and_soak(st, c, records)
        ctx["t_prod_start"] = self.heated_temperature(st)
        return ctx

    def summarise_cycle(self, st: WellState, c: CycleControls, records: list, ctx: dict):
        oil0, water0, start_day = ctx["oil0"], ctx["water0"], ctx["start_day"]
        inj, t_prod_start = ctx["inj"], ctx["t_prod_start"]
        prod = [r for r in records if r["phase"] == "production"]
        oil_obs = float(sum(r["oil_m3d"] for r in prod))
        water_obs = float(sum(r["water_m3d"] for r in prod))
        energy = float(sum(r["motor_kwh"] for r in prod))
        summary = dict(
            cycle_no=st.cycle_no, start_day=start_day, end_day=st.day,
            steam_t=c.steam_mass / 1e3, inj_pressure_MPa=c.inj_pressure / 1e6,
            quality_surface=c.quality_surface, soak_days=c.soak_days,
            cutoff_m3d=c.cutoff_rate * DAY, min_prod_days=c.min_prod_days,
            spm_policy=c.spm_policy, spm_init=c.spm_init,
            prod_days=len(prod), oil_m3=oil_obs, water_m3=water_obs,
            oil_m3_true=st.cum_oil - oil0, water_m3_true=st.cum_water - water0,
            sor=(c.steam_mass / 1e3) / max(oil_obs, 1e-9),
            peak_oil_m3d=max(r["oil_m3d"] for r in prod),
            t_bottom_start=prod[0]["t_bottom"], t_bottom_end=prod[-1]["t_bottom"],
            t_prod_start_true=t_prod_start,
            energy_kwh=energy, kwh_per_m3_oil=energy / max(oil_obs, 1e-9),
            floating_days=int(sum(r["floating"] for r in prod)),
            impact_strokes=float(sum(r["impact_strokes"] for r in prod)),
            max_goodman=max(r["goodman"] for r in prod),
            overload_days=int(sum(r["goodman"] > 1.0 for r in prod)),
            mean_spm=float(np.mean([r["spm"] for r in prod])),
            **inj,
        )
        return summary

    # ------------------------------------------------------------ economics
    @staticmethod
    def cycle_value(summary, econ):
        """Net value (USD) of a cycle and per cycle-day - used by the optimisers."""
        revenue = summary["oil_m3"] / BBL * econ["oil_price"]
        cost = (summary["fuel_GJ"] * econ["fuel_cost_per_GJ"]
                + summary["energy_kwh"] * econ["power_cost"])
        days = summary["end_day"] - summary["start_day"] + 1
        cost += days * econ["opex_per_day"]
        return revenue - cost, (revenue - cost) / days


def economics_from_config(cfg):
    return {k: cfg.value(f"economics.{k}") for k in ("oil_price", "fuel_cost_per_GJ", "power_cost",
                                                     "opex_per_day")}
