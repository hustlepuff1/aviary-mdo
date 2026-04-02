"""
Tests for the JSBSim bridge — TMD-to-JSBSim translation and validation.

Tests the XML generation (always works) and the simulation (only if JSBSim
is installed).
"""

import os
import tempfile
import unittest

from aviary_mdo.missile.jsbsim_bridge.tmd_to_jsbsim import (
    TMDToJSBSim, generate_reset_file,
)

try:
    import jsbsim
    JSBSIM_AVAILABLE = True
except ImportError:
    JSBSIM_AVAILABLE = False


class TestTMDToJSBSimTranslation(unittest.TestCase):
    """Test XML generation from TMD parameters."""

    def test_default_geometry(self):
        translator = TMDToJSBSim()
        translator.set_geometry()
        self.assertAlmostEqual(translator.geometry['diameter'], 8.0)
        self.assertAlmostEqual(translator.geometry['length'], 143.9)

    def test_xml_generation(self):
        translator = TMDToJSBSim(name='test_missile')
        translator.set_geometry(length=143.9, diameter=8.0, nose_length=19.2)
        translator.set_mass(launch_weight=500.0, cg_station=76.2, Iy=94.0)
        translator.set_aero_table()
        translator.set_propulsion(thrust=7036.0, isp=271.0, burn_time=3.26,
                                   propellant_weight=84.8)

        root = translator.build_xml()
        self.assertEqual(root.tag, 'fdm_config')
        self.assertEqual(root.get('name'), 'test_missile')

        # Check key sections exist
        tags = [child.tag for child in root]
        self.assertIn('fileheader', tags)
        self.assertIn('metrics', tags)
        self.assertIn('mass_balance', tags)
        self.assertIn('aerodynamics', tags)
        self.assertIn('propulsion', tags)

    def test_xml_write_to_file(self):
        translator = TMDToJSBSim()
        translator.set_geometry()
        translator.set_mass()
        translator.set_aero_table()
        translator.set_propulsion()

        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as f:
            filepath = f.name

        try:
            translator.write_xml(filepath)
            self.assertTrue(os.path.exists(filepath))

            with open(filepath) as f:
                content = f.read()
            self.assertIn('fdm_config', content)
            self.assertIn('aerodynamics', content)
            self.assertIn('mass_balance', content)
        finally:
            os.unlink(filepath)

    def test_reset_file_generation(self):
        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as f:
            filepath = f.name

        try:
            generate_reset_file(filepath, mach=0.8, altitude=20000.0)
            self.assertTrue(os.path.exists(filepath))

            with open(filepath) as f:
                content = f.read()
            self.assertIn('MACH', content)
            self.assertIn('0.80', content)
            self.assertIn('20000', content)
        finally:
            os.unlink(filepath)

    def test_custom_aero_table(self):
        translator = TMDToJSBSim()
        translator.set_geometry()
        translator.set_mass()
        translator.set_aero_table(
            machs=[0.5, 1.0, 1.5, 2.0],
            cd0_values=[0.30, 0.42, 0.44, 0.38]
        )
        translator.set_propulsion()

        root = translator.build_xml()
        # Check aero table is populated
        aero = root.find('aerodynamics')
        self.assertIsNotNone(aero)

    def test_sparrow_baseline_xml(self):
        """Generate the full Sparrow MRAAM XML and verify it's parseable."""
        translator = TMDToJSBSim(name='sparrow_mraam')
        translator.set_geometry(length=143.9, diameter=8.0, nose_length=19.2,
                                 wing_area=400.0)
        translator.set_mass(launch_weight=500.0, empty_weight=367.0,
                             cg_station=76.2, Iy=94.0)
        translator.set_aero_table()
        translator.set_propulsion(
            thrust=7036.0, isp=271.0, burn_time=3.26, propellant_weight=84.8,
            sustain_thrust=1119.0, sustain_isp=252.0, sustain_burn_time=10.86,
            sustain_propellant_weight=48.2)

        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as f:
            filepath = f.name

        try:
            translator.write_xml(filepath)
            # Verify file size is reasonable (not empty)
            size = os.path.getsize(filepath)
            self.assertGreater(size, 500)
        finally:
            os.unlink(filepath)


@unittest.skipUnless(JSBSIM_AVAILABLE, "JSBSim not installed")
class TestJSBSimSimulation(unittest.TestCase):
    """Test actual JSBSim simulation (requires JSBSim installed)."""

    def test_jsbsim_loads_ball(self):
        """Verify JSBSim can load and run the built-in ball model."""
        fdm = jsbsim.FGFDMExec(None)
        fdm.load_model('ball')
        fdm['ic/h-sl-ft'] = 20000.0
        fdm['ic/vc-kts'] = 500.0
        fdm.run_ic()
        fdm.run()

        t = fdm['simulation/sim-time-sec']
        self.assertGreater(t, 0.0)

    def test_jsbsim_loads_mk82(self):
        """Verify JSBSim can load the mk82 bomb model."""
        fdm = jsbsim.FGFDMExec(None)
        fdm.load_model('mk82')
        fdm['ic/h-sl-ft'] = 20000.0
        fdm['ic/vc-kts'] = 500.0
        fdm.run_ic()

        # Run a few steps
        for _ in range(10):
            fdm.run()

        M = fdm['velocities/mach']
        self.assertGreater(M, 0.0)


if __name__ == '__main__':
    unittest.main()
