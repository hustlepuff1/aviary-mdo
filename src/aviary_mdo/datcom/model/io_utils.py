"""Input/output utilities for Missile DATCOM.

Generates for005.dat input files and parses for006.dat output files.
"""

import re
import numpy as np
from pathlib import Path


def _fmt_array(name, values, per_line=6):
    """Format a namelist array assignment."""
    vals = ','.join(f'{v}' for v in values)
    return f'         {name}={vals},'


def generate_input(mach, alpha, reynolds=None, altitude=None,
                   body=None, fins=None, xcg=None, sref=None, lref=None,
                   trim_set=None, controls=None):
    """Generate a for005.dat input string.

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
        Axisymmetric body geometry. Keys: lnose, dnose, lcentr, dexit,
        base (bool), betan, laft, daft, etc.
    fins : list of dict, optional
        Up to 4 fin sets. Each dict has: xle, npanel, phif, sweep, sta,
        chord, sspan, zupper, lmaxu, lflatu, ler, sectyp, etc.
    xcg : float, optional
        Moment center x-location
    sref : float, optional
        Reference area
    lref : float, optional
        Reference length
    trim_set : int, optional
        Fin set number for trim (triggers Case 2)
    controls : list of str, optional
        Control cards: 'PART', 'PLOT', 'PRESSURES', 'SAVE', 'DAMP', etc.

    Returns
    -------
    str
        Complete for005.dat content
    """
    lines = []

    # Flight conditions
    if isinstance(mach, (int, float)):
        mach = [mach]
    if isinstance(alpha, (int, float)):
        alpha = [alpha]

    lines.append(' $FLTCON NALPHA={:.0f}.,NMACH={:.0f}.,'.format(
        len(alpha), len(mach)))
    lines.append('         MACH={},'.format(
        ','.join(f'{m}' for m in mach)))
    if reynolds is not None:
        lines.append(f'         REN={reynolds},')
    if altitude is not None:
        lines.append(f'         ALT={altitude},')

    # Alpha array (split across lines if needed)
    for i in range(0, len(alpha), 6):
        chunk = alpha[i:i+6]
        prefix = 'ALPHA' if i == 0 else f'ALPHA({i+1})'
        lines.append('         {}={},'.format(
            prefix, ','.join(f'{a}.' if isinstance(a, int) else f'{a}'
                            for a in chunk)))
    lines[-1] = lines[-1].rstrip(',') + '$'

    # Reference quantities
    if xcg is not None or sref is not None or lref is not None:
        parts = []
        if xcg is not None:
            parts.append(f'XCG={xcg}')
        if sref is not None:
            parts.append(f'SREF={sref}')
        if lref is not None:
            parts.append(f'LREF={lref}')
        lines.append(' $REFQ {},$'.format(','.join(parts)))

    # Body geometry
    if body is not None:
        body_parts = []
        for key, val in body.items():
            if isinstance(val, bool):
                body_parts.append(f'{key.upper()}=.{str(val).upper()}.')
            else:
                body_parts.append(f'{key.upper()}={val}')
        # Split into multiple $AXIBOD if needed (max ~70 chars per line)
        current = ' $AXIBOD '
        for part in body_parts:
            if len(current + part + ',') > 70:
                lines.append(current.rstrip(',') + ',$')
                current = ' $AXIBOD '
            current += part + ','
        lines.append(current.rstrip(',') + ',$')

    # Fin sets
    if fins is not None:
        for i, fin in enumerate(fins, 1):
            fin_parts = []
            for key, val in fin.items():
                if isinstance(val, (list, tuple, np.ndarray)):
                    vstr = ','.join(str(v) for v in val)
                    fin_parts.append(f'{key.upper()}={vstr}')
                else:
                    fin_parts.append(f'{key.upper()}={val}')
            current = f' $FINSET{i} '
            for part in fin_parts:
                if len(current + part + ',') > 70:
                    lines.append(current.rstrip(',') + ',')
                    current = '          '
                current += part + ','
            lines.append(current.rstrip(',') + ',$')

    # Control cards
    if controls:
        for card in controls:
            lines.append(card)

    lines.append('NEXT CASE')

    # Trim case
    if trim_set is not None:
        lines.append(f' $TRIM SET={trim_set}.,$')
        lines.append('PRINT AERO TRIM')
        lines.append('NEXT CASE')

    return '\n'.join(lines) + '\n'


def parse_output(filepath='for006.dat'):
    """Parse a for006.dat output file.

    Returns
    -------
    dict
        Dictionary with keys per case. Each case contains:
        - 'alpha': array of angles of attack
        - 'CN', 'CM', 'CA', 'CY', 'CLN', 'CLL': aero coefficients
        - 'CNA', 'CMA', 'CYB', 'CLNB', 'CLLB': derivatives (if present)
        - 'CL', 'CD': lift/drag (if present)
        - 'mach', 'reynolds', 'sref', 'lref': flight conditions
        - 'warnings': list of warning messages
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f'{filepath} not found')

    content = path.read_text()
    cases = {}
    current_case = 0
    current_section = None

    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]

        # Case/page markers
        m = re.search(r'CASE\s+(\d+)', line)
        if m:
            current_case = int(m.group(1))
            if current_case not in cases:
                cases[current_case] = {
                    'warnings': [],
                    'sections': {}
                }

        # Warnings
        if '* WARNING *' in line:
            if current_case in cases:
                cases[current_case]['warnings'].append(line.strip())

        # Flight conditions
        m = re.search(r'MACH NO\s*=\s*([\d.E+-]+)', line)
        if m and current_case in cases:
            cases[current_case]['mach'] = float(m.group(1))

        m = re.search(r'REYNOLDS NO\s*=\s*([\d.E+-]+)', line)
        if m and current_case in cases:
            cases[current_case]['reynolds'] = float(m.group(1))

        m = re.search(r'REF AREA\s*=\s*([\d.E+-]+)', line)
        if m and current_case in cases:
            cases[current_case]['sref'] = float(m.group(1))

        m = re.search(r'REF LENGTH\s*=\s*([\d.E+-]+)', line)
        if m and current_case in cases:
            cases[current_case]['lref'] = float(m.group(1))

        # Coefficient tables - detect header row with ALPHA + known coefficients
        if re.search(r'ALPHA\s+(CN|CL|CNA|DELTA)', line):
            col_names = line.split()
            col_names = [c for c in col_names if c != 'ALPHA']

            # Read data rows
            data_rows = []
            i += 1
            while i < len(lines):
                row = lines[i].strip()
                if not row:
                    if not data_rows:
                        i += 1
                        continue
                    else:
                        break
                if row.startswith('1 ') or 'PAGE' in row:
                    break
                parts = row.split()
                try:
                    vals = []
                    for p in parts:
                        if p == '*NT*':
                            vals.append(np.nan)
                        else:
                            vals.append(float(p))
                    if len(vals) >= 2:
                        data_rows.append(vals)
                except ValueError:
                    break
                i += 1

            if data_rows and current_case in cases:
                data = np.array(data_rows)
                section_name = '_'.join(col_names[:3])
                section = {'alpha': data[:, 0]}
                for j, name in enumerate(col_names):
                    if j + 1 < data.shape[1]:
                        section[name] = data[:, j + 1]
                cases[current_case]['sections'][section_name] = section

                # Also store at top level (last table wins for each coeff)
                cases[current_case]['alpha'] = data[:, 0]
                for j, name in enumerate(col_names):
                    if j + 1 < data.shape[1]:
                        cases[current_case][name] = data[:, j + 1]

        i += 1

    return cases
