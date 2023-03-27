from chem import molecular_formula


def test_a():
    m = molecular_formula.MolecularFormula("C6SeH12UO6")
    print(m.get_mass())
    print(m.get_adduct_mass('[M+H]+'))

    m2 = molecular_formula.MolecularFormula(copy_from=m)
    print(m2)

    print(molecular_formula.calculate_mass('C6HO6'))
    print(molecular_formula.calculate_mass_with_adduct('C6HO6', '[M-H]-'))


test_a()
