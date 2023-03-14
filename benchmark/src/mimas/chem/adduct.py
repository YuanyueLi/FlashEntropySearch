from .mass import atom_mass as ATOM_MASS, single_charged_adduct_mass as COMMON_ADDUCT_MASS
import re

informal_adduct_mass = {
    "i": 1.00866491588,
    "IsoProp": 60.05751,
    "ACN": 41.02655,
    "Hac": 60.02113,
    "HAc": 60.02113,
    "FA": 46.00548,
    "TFA": 113.99286,
    "DMSO": 78.01394,

    "Formate": 44.99765,
    "formate": 44.99765,
    "MeOH": 30.01056,
    "Oac": 60.02113,
    "OAc": 60.02113,
    "acetate": 60.02113,
    "HFA": 165.98533,
}


def get_adduct_mass(formula):
    adduct_mol_num, formula_str = re.findall("(^\d*)(.*$)", formula)[0]
    if adduct_mol_num == "":
        adduct_mol_num = 1
    else:
        adduct_mol_num = int(adduct_mol_num)

    if formula_str in informal_adduct_mass:
        mol_mass = informal_adduct_mass[formula_str]
    else:
        all_atom_nums = re.findall('([A-Z][a-z]*)([0-9]*)', formula_str)
        mol_mass = 0.
        for atom_num in all_atom_nums:
            n = atom_num[1]
            if n == '':
                mol_mass += ATOM_MASS[atom_num[0]]
            else:
                mol_mass += int(n) * ATOM_MASS[atom_num[0]]
    return adduct_mol_num*mol_mass


class Adduct:
    def __init__(self, adduct: str):
        self.correct = True
        try:
            charge = {'+': 1, '-': -1}[adduct[-1]]
        except:
            self.correct = False
            return

        if adduct in COMMON_ADDUCT_MASS:
            self.mass_add = COMMON_ADDUCT_MASS[adduct]
            self.m_num = 1
            self.charge = charge
        else:
            E_MASS = 0.00054858
            try:
                m_num_str, mol_add_str, charge_str = re.findall(r"\[(\d*)M(.*)\](\d*)[+-]", adduct)[0]
                if m_num_str:
                    self.m_num = int(m_num_str)
                else:
                    self.m_num = 1
                if charge_str:
                    self.charge = int(charge_str) * charge
                else:
                    self.charge = charge
                self.mass_add = 0.
                for mol in re.findall("([+-])([^+-]+)", mol_add_str):
                    if mol[0] == '+':
                        self.mass_add += get_adduct_mass(mol[1])
                    else:
                        self.mass_add -= get_adduct_mass(mol[1])
                self.mass_add -= self.charge*E_MASS
            except:
                print(f"Error in parsing adduct {adduct}")
                self.charge = charge
                self.m_num = 1
                self.mass_add = 0.
                self.correct = False

    def get_mz_with_adduct(self, monoisotopic_mass):
        mz = (monoisotopic_mass*self.m_num + self.mass_add)/abs(self.charge)
        return mz

    def get_mass_from_mz_with_adduct(self, mz):
        mass = (mz*abs(self.charge)-self.mass_add)/self.m_num
        return mass

    def simplify_adduct(self, mz):
        """
        This function will remove all the adducts first.
        Then add [M+H]+ for +1, [M-H]- for -1, [M+2H]+ for +2, [M-2H]- for -2, etc.
        """
        mass = self.get_mass_from_mz_with_adduct(mz)
        mz = (mass + COMMON_ADDUCT_MASS['[M+H]+']*self.charge)/abs(self.charge)
        return mz


if __name__ == '__main__':
    print(Adduct('[M+2H+Na]3+').get_mz_with_adduct(853.33089))
    print(Adduct('[2M+ACN+Na]+').get_mz_with_adduct(853.33089))
    print(Adduct('[M+DMSO+H]+').get_mz_with_adduct(853.33089))
    print(Adduct('[M+IsoProp+Na+H]+').get_mz_with_adduct(853.33089))
    print(Adduct('[M+CH3OH+H]+').get_mz_with_adduct(853.33089))
    print(Adduct('[M+2ACN+H]+').get_mz_with_adduct(853.33089))
    print(Adduct('[M+3H]3+').get_mz_with_adduct(853.33089))
    print(Adduct('[M-3H]3-').get_mz_with_adduct(853.33089))
    print(Adduct('[2M+FA-H]-').get_mz_with_adduct(853.33089))
    print(Adduct('[M-H]-').get_mz_with_adduct(853.33089))
    print(Adduct('[M-H]-').get_mz_with_adduct(853.33089))
