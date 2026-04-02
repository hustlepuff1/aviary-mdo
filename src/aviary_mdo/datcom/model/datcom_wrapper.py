"""Main wrapper class for Missile DATCOM.

Provides a Pythonic interface to the Fortran executable.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .io_utils import generate_input, parse_output


# Default location of the Missile DATCOM installation
_DEFAULT_DATCOM_DIR = Path(os.environ.get(
    'MISSILE_DATCOM_DIR',
    Path.home() / 'MDO' / 'Missile-Datcom'
))


def find_executable(datcom_dir=None):
    """Locate the misdat executable.

    Searches (in order):
    1. Explicit datcom_dir / misdat
    2. $MISSILE_DATCOM_DIR / misdat
    3. ~/MDO/Missile-Datcom/misdat
    4. ~/MDO/Missile-Datcom/Source-F90/misdat

    Returns
    -------
    Path
        Resolved path to the misdat executable.

    Raises
    ------
    FileNotFoundError
        If no executable is found.
    """
    base = Path(datcom_dir) if datcom_dir else _DEFAULT_DATCOM_DIR
    candidates = [
        base / 'misdat',
        base / 'Source-F90' / 'misdat',
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        f'misdat executable not found in {base}. '
        'Build with: cd "Source Code-Missile DATCOM" && '
        'ifx -fixed -w *.f -o ../misdat')


def find_finder_dat(datcom_dir=None):
    """Locate FINDER.DAT (binary lookup tables).

    Returns
    -------
    Path
        Resolved path to FINDER.DAT.
    """
    base = Path(datcom_dir) if datcom_dir else _DEFAULT_DATCOM_DIR
    path = base / 'FINDER.DAT'
    if not path.exists():
        raise FileNotFoundError(f'FINDER.DAT not found in {base}')
    return path.resolve()


class MissileDATCOM:
    """Python wrapper for the USAF Missile DATCOM aerodynamic prediction code.

    Parameters
    ----------
    datcom_dir : str or Path, optional
        Path to the Missile-Datcom installation directory.
        Defaults to $MISSILE_DATCOM_DIR or ~/MDO/Missile-Datcom.
    exe_path : str or Path, optional
        Explicit path to the misdat executable (overrides datcom_dir search).
    keep_files : bool
        If True, don't clean up I/O files after run.

    Examples
    --------
    >>> md = MissileDATCOM()
    >>> results = md.run(
    ...     mach=[2.36],
    ...     alpha=[0, 4, 8, 12, 16, 20, 24, 28],
    ...     reynolds=3e6,
    ...     body={'lnose': 11.25, 'dnose': 3.75, 'lcentr': 26.25, 'dexit': 2.0},
    ...     xcg=18.75
    ... )
    >>> print(results[1]['CN'])
    """

    def __init__(self, datcom_dir=None, exe_path=None, keep_files=False):
        self.keep_files = keep_files

        if exe_path:
            self.exe_path = Path(exe_path).resolve()
        else:
            self.exe_path = find_executable(datcom_dir)

        self.finder_path = find_finder_dat(datcom_dir)

    def run(self, mach, alpha, reynolds=None, altitude=None,
            body=None, fins=None, xcg=None, sref=None, lref=None,
            trim_set=None, controls=None, input_file=None):
        """Run Missile DATCOM analysis.

        Parameters
        ----------
        mach : float or list
            Mach number(s)
        alpha : list
            Angles of attack (degrees)
        reynolds : float, optional
            Reynolds number per foot
        altitude : float, optional
            Altitude in feet
        body : dict, optional
            Axisymmetric body geometry
        fins : list of dict, optional
            Up to 4 fin set definitions
        xcg : float, optional
            Moment center x-location
        sref : float, optional
            Reference area
        lref : float, optional
            Reference length
        trim_set : int, optional
            Fin set for trim analysis
        controls : list of str, optional
            Control cards (PART, PLOT, etc.)
        input_file : str, optional
            Path to pre-built for005.dat (overrides other params)

        Returns
        -------
        dict
            Results keyed by case number. Each case has arrays:
            alpha, CN, CM, CA, CY, CLN, CLL, etc.
        """
        work = Path(tempfile.mkdtemp(prefix='datcom_'))
        cleanup = not self.keep_files

        try:
            # Copy FINDER.DAT to working directory
            shutil.copy2(self.finder_path, work / 'FINDER.DAT')

            # Generate or copy input file
            if input_file:
                shutil.copy2(input_file, work / 'for005.dat')
            else:
                if controls is None:
                    controls = ['PART']
                input_text = generate_input(
                    mach=mach, alpha=alpha, reynolds=reynolds,
                    altitude=altitude, body=body, fins=fins,
                    xcg=xcg, sref=sref, lref=lref,
                    trim_set=trim_set, controls=controls)
                (work / 'for005.dat').write_text(input_text)

            # Run the executable
            result = subprocess.run(
                [str(self.exe_path)],
                cwd=str(work),
                capture_output=True, text=True,
                timeout=60)

            if result.returncode != 0:
                raise RuntimeError(
                    f'DATCOM failed (rc={result.returncode}): {result.stderr}')

            # Parse output
            output = parse_output(str(work / 'for006.dat'))
            output['_work_dir'] = str(work)

            return output

        finally:
            if cleanup:
                shutil.rmtree(work, ignore_errors=True)

    def run_input_file(self, input_path):
        """Run DATCOM with an existing for005.dat file."""
        return self.run(mach=0, alpha=[0], input_file=input_path)
