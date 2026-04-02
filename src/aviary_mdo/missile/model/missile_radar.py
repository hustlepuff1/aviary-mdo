"""
Tactical Missile Design radar detection components.

Implements the TMD Radar sheet (sheet 9) equations:
  - Radar range equation for missile tracking target
  - Threat tracking missile detection range
  - Detection range vs RCS table
  - Visual detection range

Radar range equation (monostatic):
  R = ((Pt * G^2 * lambda^2 * sigma * n) /
       ((4*pi)^3 * k * T_rec * Bn * Fn * SNR * L))^(1/4)

where:
  G   = antenna gain = eta * (pi * D / lambda)^2, eta = 0.5625
  Pt  = transmitter peak power (W)
  lambda = radar wavelength (m)
  sigma = target radar cross section (m^2)
  n   = number of pulses integrated
  k   = Boltzmann constant (1.38064852e-23 J/K)
  T_rec = receiver noise temperature (K)
  Bn  = noise bandwidth (Hz)
  Fn  = receiver noise factor
  SNR = required signal-to-noise ratio
  L   = system loss factor

Reference: Fleeman, E., "Tactical Missile Design," AIAA, 2001, Chapter 9.
"""

import numpy as np
import openmdao.api as om


# Boltzmann constant (J/K)
_BOLTZMANN = 1.38064852e-23

# Aperture efficiency for uniformly illuminated circular aperture
_ETA_APERTURE = 0.5625


class MissileRadar(om.ExplicitComponent):
    """
    Missile seeker radar detection range calculation.

    Computes:
      - 3-dB beamwidth of the antenna
      - Missile-tracking-target detection range (monostatic radar equation)
      - Threat-tracking-missile detection range (same equation, missile RCS, SNR=1)
      - Detection range vs RCS table for selected RCS values
    """

    def setup(self):
        # Radar parameters
        self.add_input('radar_wavelength', val=0.03, units='m',
                       desc='Radar wavelength')
        self.add_input('antenna_diameter', val=10.0, units='inch',
                       desc='Antenna diameter')
        self.add_input('transmitter_power', val=1000.0, units='W',
                       desc='Transmitter peak power')
        self.add_input('frequency', val=10.0e9, units='Hz',
                       desc='Radar frequency')

        # Receiver parameters
        self.add_input('receiver_temperature', val=290.0, units='K',
                       desc='Receiver noise temperature')
        self.add_input('noise_bandwidth', val=1.0e6, units='Hz',
                       desc='Receiver noise bandwidth')
        self.add_input('loss_factor', val=5.0,
                       desc='System loss factor (linear)')
        self.add_input('num_pulses_integrated', val=100.0,
                       desc='Number of pulses integrated')
        self.add_input('receiver_noise_factor', val=5.0,
                       desc='Receiver noise factor (linear)')
        self.add_input('signal_to_noise_ratio', val=10.0,
                       desc='Required signal-to-noise ratio (linear)')

        # Target parameters
        self.add_input('missile_rcs', val=0.01, units='m**2',
                       desc='Missile radar cross section')
        self.add_input('target_rcs', val=10.0, units='m**2',
                       desc='Target radar cross section')

        # Outputs
        self.add_output('beamwidth', val=0.0, units='rad',
                        desc='Antenna 3-dB beamwidth')
        self.add_output('antenna_gain', val=0.0,
                        desc='Antenna gain (linear)')
        self.add_output('detection_range', val=0.0, units='m',
                        desc='Missile tracking target detection range')
        self.add_output('detection_range_nmi', val=0.0, units='nmi',
                        desc='Detection range in nautical miles')
        self.add_output('detection_range_ft', val=0.0, units='ft',
                        desc='Detection range in feet')
        self.add_output('threat_range', val=0.0, units='m',
                        desc='Threat tracking missile detection range')

        # Detection range vs RCS table outputs
        self._rcs_values = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
        for i, rcs in enumerate(self._rcs_values):
            self.add_output(f'det_range_rcs_{i}', val=0.0, units='m',
                            desc=f'Detection range for RCS={rcs} m^2')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        lam = inputs['radar_wavelength']
        d_in = inputs['antenna_diameter']
        Pt = inputs['transmitter_power']
        T_rec = inputs['receiver_temperature']
        Bn = inputs['noise_bandwidth']
        L = inputs['loss_factor']
        n_pulses = inputs['num_pulses_integrated']
        Fn = inputs['receiver_noise_factor']
        SNR = inputs['signal_to_noise_ratio']
        sigma_missile = inputs['missile_rcs']
        sigma_target = inputs['target_rcs']

        # Convert antenna diameter to meters
        d_m = d_in * 0.0254

        # 3-dB beamwidth for uniformly illuminated circular aperture
        beamwidth = 1.02 * lam / d_m
        outputs['beamwidth'] = beamwidth

        # Antenna gain
        G = _ETA_APERTURE * (np.pi * d_m / lam) ** 2
        outputs['antenna_gain'] = G

        # Radar range equation common denominator factor
        denom_common = ((4.0 * np.pi) ** 3 * _BOLTZMANN * T_rec
                        * Bn * Fn * L)

        # Missile tracking target detection range
        num_target = Pt * G ** 2 * lam ** 2 * sigma_target * n_pulses
        den_target = denom_common * SNR
        R_target = (num_target / den_target) ** 0.25
        outputs['detection_range'] = R_target
        outputs['detection_range_nmi'] = R_target / 1852.0
        outputs['detection_range_ft'] = R_target * 3.28084

        # Threat tracking missile detection range (SNR = 1)
        num_threat = Pt * G ** 2 * lam ** 2 * sigma_missile * n_pulses
        den_threat = denom_common * 1.0
        R_threat = (num_threat / den_threat) ** 0.25
        outputs['threat_range'] = R_threat

        # Detection range vs RCS table
        for i, rcs in enumerate(self._rcs_values):
            num_rcs = Pt * G ** 2 * lam ** 2 * rcs * n_pulses
            den_rcs = denom_common * SNR
            R_rcs = (num_rcs / den_rcs) ** 0.25
            outputs[f'det_range_rcs_{i}'] = R_rcs


class VisualDetectionRange(om.ExplicitComponent):
    """
    Visual detection range estimate.

    R_visual = Ct * sqrt(A_target)

    where Ct is a visual constant (typically 1500-2500 m/m for clear air)
    and A_target is the target presented area in m^2.
    """

    def setup(self):
        self.add_input('target_presented_area', val=1.0, units='m**2',
                       desc='Target presented area')
        self.add_input('visual_constant', val=2000.0,
                       desc='Visual detection constant (m per sqrt(m^2))')

        self.add_output('visual_range', val=0.0, units='m',
                        desc='Visual detection range')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        A = inputs['target_presented_area']
        Ct = inputs['visual_constant']

        outputs['visual_range'] = Ct * np.sqrt(A)
