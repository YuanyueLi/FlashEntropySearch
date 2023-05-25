import numpy as np
import re
from . import mass
import itertools
import math

_numpy_formula_format = np.int16

_atom_dict = {a: i for i, a in enumerate(mass.atom_mass)}
_len_atom_dict = len(_atom_dict)
_atom_mass_array = np.zeros(_len_atom_dict, np.float32)
for a in _atom_dict:
    _atom_mass_array[_atom_dict[a]] = mass.atom_mass[a]


def calculate_mass(formula: str):
    all_atom_nums = re.findall('([A-Z][a-z]*)([0-9]*)', formula)
    mol_mass = 0.
    try:
        for atom_num in all_atom_nums:
            n = atom_num[1]
            if n == '':
                mol_mass += mass.atom_mass[atom_num[0]]
            else:
                mol_mass += int(n) * mass.atom_mass[atom_num[0]]
    except KeyError as e:
        print("Atom {} is not known".format(e.args[0]))
    return mol_mass


def calculate_mass_with_adduct(formula: str, adduct: str):
    return calculate_mass(formula) + mass.adduct_ions_mass[adduct]


class MolecularFormula(object):
    __slots__ = ['_data', '_hash']

    def __init__(self, formula: str = None, data=None, copy_from: 'MolecularFormula' = None):
        self._data = np.zeros(_len_atom_dict, _numpy_formula_format)
        if formula is not None:
            self.from_string(key_string=formula)
        if data is not None:
            self._data = np.array(data, _numpy_formula_format, copy=True)
        if copy_from is not None:
            self._data = np.array(copy_from.get_data(), _numpy_formula_format, copy=True)
        self._hash = None

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(str(self))
        return self._hash

    def __getitem__(self, item):
        return self._data[_atom_dict[item]]

    def __setitem__(self, key, value):
        self._data[_atom_dict[key]] = value

    def __str__(self):
        string = ''

        for atom in mass.atom_mass:
            atom_num = self[atom]
            if atom_num:
                if atom_num > 1:
                    string += atom + str(atom_num)
                else:
                    string += atom
        return string

    def from_string(self, key_string):
        self._hash = None
        all_atom_nums = re.findall('([A-Z][a-z]*)([0-9]*)', key_string)
        for atom_num in all_atom_nums:
            n = atom_num[1]
            if n == '':
                self[atom_num[0]] = 1
            else:
                self[atom_num[0]] = int(n)

    def get_data(self):
        return self._data

    def get_mass(self):
        return np.sum(_atom_mass_array * self._data)

    def get_adduct_mass(self, adduct):
        return self.get_mass() + mass.adduct_ions_mass[adduct]

    def get_degree_of_unsaturation(self) -> float:
        x_num = self["F"] + self["Cl"] + self["Br"] + self["I"]
        h_num = self["H"] + self["D"] + self["T"]
        dou = (self["C"] * 2 + 2 + self["N"] - x_num - h_num) / 2
        return dou


def _calculate_formula(mass_start, mass_end, candidate_formula_array, cur_i, result):
    atom_mass_cur = _atom_mass_array[cur_i]
    atom_num = math.floor(mass_end / atom_mass_cur)
    if cur_i == 0:
        # This is H
        h_num_low = mass_start / atom_mass_cur
        if atom_num >= h_num_low:
            candidate_formula_array[0] = atom_num
            result.append(MolecularFormula(candidate_formula_array))
    else:
        for i in range(atom_num):
            f = np.copy(candidate_formula_array)
            f[cur_i] = i
            _calculate_formula(mass_start - i * atom_mass_cur, mass_end - i * atom_mass_cur,
                               f, cur_i - 1, result)


def precursor_mass_to_formula(lo_mass: float, hi_mass: float, adduct: str):
    lo_mass -= mass.adduct_ions_mass[adduct]
    hi_mass -= mass.adduct_ions_mass[adduct]

    result = []
    candidate_formula_array = np.zeros(_len_atom_dict, _numpy_formula_format)
    _calculate_formula(lo_mass, hi_mass, candidate_formula_array,
                       len(candidate_formula_array) - 1, result)
    return result


def product_mass_to_formula(lo_mass: float, hi_mass: float, adduct: str, precursor_formula):
    lo_mass -= mass.adduct_ions_mass[adduct]
    hi_mass -= mass.adduct_ions_mass[adduct]

    # Generate candidate range
    precursor_data = precursor_formula.get_data()
    formula_range = [range(x + 1) for x in precursor_data]
    all_possible_candidate_formula = np.array(list(itertools.product(*formula_range)), _numpy_formula_format)
    all_possible_mass = np.sum(_atom_mass_array * all_possible_candidate_formula, axis=1)

    candidate_data = all_possible_candidate_formula[(lo_mass <= all_possible_mass) & (all_possible_mass <= hi_mass)]

    result = []
    for data in candidate_data:
        formula = MolecularFormula(data=data)
        result.append(formula)
    return result
