# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Information of chemical elements
"""
#pylint: disable=bad-whitespace

import numpy as np

elements = np.array([
    '',
    'H',
    'He',
    'Li',
    'Be',
    'B',
    'C',
    'N',
    'O',
    'F',
    'Ne',
    'Na',
    'Mg',
    'Al',
    'Si',
    'P',
    'S',
    'Cl',
    'Ar',
    'K',
    'Ca',
    'Sc',
    'Ti',
    'V',
    'Cr',
    'Mn',
    'Fe',
    'Co',
    'Ni',
    'Cu',
    'Zn',
    'Ga',
    'Ge',
    'As',
    'Se',
    'Br',
    'Kr',
    'Rb',
    'Sr',
    'Y',
    'Zr',
    'Nb',
    'Mo',
    'Tc',
    'Ru',
    'Rh',
    'Pd',
    'Ag',
    'Cd',
    'In',
    'Sn',
    'Sb',
    'Te',
    'I',
    'Xe',
    'Cs',
    'Ba',
    'La',
    'Ce',
    'Pr',
    'Nd',
    'Pm',
    'Sm',
    'Eu',
    'Gd',
    'Tb',
    'Dy',
    'Ho',
    'Er',
    'Tm',
    'Yb',
    'Lu',
    'Hf',
    'Ta',
    'W',
    'Re',
    'Os',
    'Ir',
    'Pt',
    'Au',
    'Hg',
    'Tl',
    'Pb',
    'Bi',
    'Po',
    'At',
    'Rn',
    'Fr',
    'Ra',
    'Ac',
    'Th',
    'Pa',
    'U',
    'Np',
    'Pu',
    'Am',
    'Cm',
    'Bk',
    'Cf',
    'Es',
    'Fm',
    'Md',
    'No',
    'Lr',
    'Rf',
    'Db',
    'Sg',
    'Bh',
    'Hs',
    'Mt',
    'Ds',
    'Rg',
    'Cn',
    'Nh',
    'Fl',
    'Mc',
    'Lv',
    'Ts',
    'Og',
])

element_set = set(elements)

element_dict = {
    'X':  0,
    '':   0,
    'H':  1,
    'He': 2,
    'Li': 3,
    'Be': 4,
    'B':  5,
    'C':  6,
    'N':  7,
    'O':  8,
    'F':  9,
    'Ne': 10,
    'Na': 11,
    'Mg': 12,
    'Al': 13,
    'Si': 14,
    'P':  15,
    'S':  16,
    'Cl': 17,
    'Ar': 18,
    'K':  19,
    'Ca': 20,
    'Sc': 21,
    'Ti': 22,
    'V':  23,
    'Cr': 24,
    'Mn': 25,
    'Fe': 26,
    'Co': 27,
    'Ni': 28,
    'Cu': 29,
    'Zn': 30,
    'Ga': 31,
    'Ge': 32,
    'As': 33,
    'Se': 34,
    'Br': 35,
    'Kr': 36,
    'Rb': 37,
    'Sr': 38,
    'Y':  39,
    'Zr': 40,
    'Nb': 41,
    'Mo': 42,
    'Tc': 43,
    'Ru': 44,
    'Rh': 45,
    'Pd': 46,
    'Ag': 47,
    'Cd': 48,
    'In': 49,
    'Sn': 50,
    'Sb': 51,
    'Te': 52,
    'I':  53,
    'Xe': 54,
    'Cs': 55,
    'Ba': 56,
    'La': 57,
    'Ce': 58,
    'Pr': 59,
    'Nd': 60,
    'Pm': 61,
    'Sm': 62,
    'Eu': 63,
    'Gd': 64,
    'Tb': 65,
    'Dy': 66,
    'Ho': 67,
    'Er': 68,
    'Tm': 69,
    'Yb': 70,
    'Lu': 71,
    'Hf': 72,
    'Ta': 73,
    'W':  74,
    'Re': 75,
    'Os': 76,
    'Ir': 77,
    'Pt': 78,
    'Au': 79,
    'Hg': 80,
    'Tl': 81,
    'Pb': 82,
    'Bi': 83,
    'Po': 84,
    'At': 85,
    'Rn': 86,
    'Fr': 87,
    'Ra': 88,
    'Ac': 89,
    'Th': 90,
    'Pa': 91,
    'U':  92,
    'Np': 93,
    'Pu': 94,
    'Am': 95,
    'Cm': 96,
    'Bk': 97,
    'Cf': 98,
    'Es': 99,
    'Fm': 100,
    'Md': 101,
    'No': 102,
    'Lr': 103,
    'Rf': 104,
    'Db': 105,
    'Sg': 106,
    'Bh': 107,
    'Hs': 108,
    'Mt': 109,
    'Ds': 110,
    'Rg': 111,
    'Cn': 112,
    'Nh': 113,
    'Fl': 114,
    'Mc': 115,
    'Lv': 116,
    'Ts': 117,
    'Og': 118,
}

element_name = np.array([
    'None',
    'Hydrogen',
    'Helium',
    'Lithium',
    'Beryllium',
    'Boron',
    'Carbon',
    'Nitrogen',
    'Oxygen',
    'Fluorine',
    'Neon',
    'Sodium',
    'Magnesium',
    'Aluminium',
    'Silicon',
    'Phosphorus',
    'Sulfur',
    'Chlorine',
    'Argon',
    'Potassium',
    'Calcium',
    'Scandium',
    'Titanium',
    'Vanadium',
    'Chromium',
    'Manganese',
    'Iron',
    'Cobalt',
    'Nickel',
    'Copper',
    'Zinc',
    'Gallium',
    'Germanium',
    'Arsenic',
    'Selenium',
    'Bromine',
    'Krypton',
    'Rubidium',
    'Strontium',
    'Yttrium',
    'Zirconium',
    'Niobium',
    'Molybdenum',
    'Technetium',
    'Ruthenium',
    'Rhodium',
    'Palladium',
    'Silver',
    'Cadmium',
    'Indium',
    'Tin',
    'Antimony',
    'Tellurium',
    'Iodine',
    'Xenon',
    'Cesium',
    'Barium',
    'Lanthanum',
    'Cerium',
    'Praseodymium',
    'Neodymium',
    'Promethium',
    'Samarium',
    'Europium',
    'Gadolinium',
    'Terbium',
    'Dysprosium',
    'Holmium',
    'Erbium',
    'Thulium',
    'Ytterbium',
    'Lutetium',
    'Hafnium',
    'Tantalum',
    'Tungsten',
    'Rhenium',
    'Osmium',
    'Iridium',
    'Platinum',
    'Gold',
    'Mercury',
    'Thallium',
    'Lead',
    'Bismuth',
    'Polonium',
    'Astatine',
    'Radon',
    'Francium',
    'Radium',
    'Actinium',
    'Thorium',
    'Protactinium',
    'Uranium',
    'Neptunium',
    'Plutonium',
    'Americium',
    'Curium',
    'Berkelium',
    'Californium',
    'Einsteinium',
    'Fermium',
    'Mendelevium',
    'Nobelium',
    'Lawrencium',
    'Rutherfordium',
    'Dubnium',
    'Seaborgium',
    'Bohrium',
    'Hassium',
    'Meitnerium',
    'Darmstadtium',
    'Roentgenium',
    'Copernicium',
    'Nihonium',
    'Flerovium',
    'Moscovium',
    'Livermorium',
    'Tennessine',
    'Oganesson',
])

atomic_mass = np.array([
    0.000,
    1.008,
    4.003,
    6.941,
    9.012,
    10.81,
    12.01,
    14.01,
    16.00,
    19.00,
    20.18,
    22.99,
    24.31,
    26.98,
    28.09,
    30.97,
    32.07,
    35.45,
    39.95,
    39.10,
    40.08,
    44.96,
    47.87,
    50.94,
    52.00,
    54.94,
    55.85,
    58.93,
    58.69,
    63.55,
    65.38,
    69.72,
    72.64,
    74.92,
    78.97,
    79.90,
    83.80,
    85.47,
    87.62,
    88.91,
    91.22,
    92.91,
    95.95,
    98.91,
    101.07,
    102.91,
    106.42,
    107.87,
    112.41,
    114.82,
    118.71,
    121.76,
    127.60,
    126.90,
    131.29,
    132.91,
    137.33,
    138.91,
    140.12,
    140.91,
    144.24,
    144.90,
    150.36,
    151.96,
    157.25,
    158.93,
    162.50,
    164.93,
    167.26,
    168.93,
    173.05,
    174.97,
    178.49,
    180.95,
    183.84,
    186.21,
    190.23,
    192.22,
    195.08,
    196.97,
    200.59,
    204.38,
    207.20,
    208.98,
    208.98,
    209.99,
    222.02,
    223.02,
    226.02,
    227.03,
    232.04,
    231.04,
    238.03,
    237.05,
    239.06,
    243.06,
    247.07,
    247.07,
    251.08,
    252.08,
    257.06,
    258.10,
    259.10,
    262.11,
    267.12,
    268.13,
    269.13,
    274.14,
    277.15,
    278.00,
    281.00,
    282.00,
    285.00,
    284.00,
    289.00,
    288.00,
    292.00,
    294.00,
    295.00,
])
