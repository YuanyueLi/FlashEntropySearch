from mimas.file_io import spec_file


def test_read(file_spec):
    for spec in spec_file.read_one_spectrum(file_spec):
        if spec["ms_level"] == 2:
            print(spec)
            break


if __name__ == '__main__':
    test_read(r"D:\test\spectra_example\MB_Qtof_C18_pos_1.mgf")
    test_read(r"D:\test\spectra_example\Neg_HILIC_45_Mix_9_4.mzML")
    test_read(r"D:\test\spectra_example\MB_Qtof_C18_pos_2.mzML")
    test_read(r"D:\test\spectra_example\Neg_HILIC_45_Mix_9_4.mzML.gz")
    test_read(r"D:\test\spectra_example\MB_Qtof_C18_pos_2.mzML.gz")
