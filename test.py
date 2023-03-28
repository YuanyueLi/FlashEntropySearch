import numpy as np
import unittest
import tempfile
from ms_entropy.flash_entropy_search import FlashEntropySearch


class TestFlashEntropySearchWithCpu(unittest.TestCase):
    def setUp(self):
        spectral_library = [
            {
                "id": "Demo spectrum 1",
                "precursor_mz": 150.0,
                "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [102.0, 1.0], [103.0, 1.0]], dtype=np.float32)
            },
            {
                "id": "Demo spectrum 2",
                "precursor_mz": 220.0,
                "peaks": np.array([[200.0, 1.0], [101.0, 1.0], [202.0, 1.0], [204.0, 1.0], [205.0, 1.0]], dtype=np.float32)
            },
            {
                "id": "Demo spectrum 3",
                "precursor_mz": 250.0,
                "peaks": np.array([[100.0, 1.0], [201.0, 1.0], [202.0, 1.0], [104.0, 1.0], [105.0, 1.0]], dtype=np.float32)
            },
            {
                "id": "Demo spectrum 4",
                "precursor_mz": 350.0,
                "peaks": [[100.0, 1.0], [101.0, 1.0], [302.0, 1.0], [104.0, 1.0], [105.0, 1.0]]
            }
        ]
        query_spectrum = {
            "precursor_mz": 150.0,
            "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [102.0, 1.0], [103.0, 1.0]], dtype=np.float32)
        }

        self.flash_entropy = FlashEntropySearch()
        self.flash_entropy.build_index(spectral_library)
        query_spectrum['peaks'] = self.flash_entropy.clean_spectrum_for_search(
            precursor_mz=query_spectrum['precursor_mz'], peaks=query_spectrum['peaks'])
        self.query_spectrum = query_spectrum
    
    def test_read_and_write(self):
        path_test = tempfile.mkdtemp()
        self.flash_entropy.write(path_test)
        self.flash_entropy.read(path_test)

    def test_hybrid_search(self):
        similarity = self.flash_entropy.hybrid_search(precursor_mz=self.query_spectrum['precursor_mz'],
                                                      peaks=self.query_spectrum['peaks'],
                                                      ms2_tolerance_in_da=0.02)
        np.testing.assert_almost_equal(similarity, [1.0, 0.22299, 0.66897, 0.66897], decimal=5)

    def test_neutral_loss_search(self):
        similarity = self.flash_entropy.neutral_loss_search(precursor_mz=self.query_spectrum['precursor_mz'],
                                                            peaks=self.query_spectrum['peaks'],
                                                            ms2_tolerance_in_da=0.02)
        np.testing.assert_almost_equal(similarity, [1.0, 0.0, 0.44598, 0.22299], decimal=5)

    def test_open_search(self):
        similarity = self.flash_entropy.open_search(peaks=self.query_spectrum['peaks'], ms2_tolerance_in_da=0.02)
        np.testing.assert_almost_equal(similarity, [1.0, 0.22299, 0.22299, 0.44598], decimal=5)

    def test_identity_search(self):
        similarity = self.flash_entropy.identity_search(precursor_mz=self.query_spectrum['precursor_mz'],
                                                        peaks=self.query_spectrum['peaks'],
                                                        ms1_tolerance_in_da=0.01, ms2_tolerance_in_da=0.02)
        np.testing.assert_almost_equal(similarity, [1.0, 0.0, 0.0, 0.0], decimal=5)


class TestFlashEntropySearchWithGpu(TestFlashEntropySearchWithCpu):
    def test_hybrid_search(self):
        similarity = self.flash_entropy.hybrid_search(precursor_mz=self.query_spectrum['precursor_mz'],
                                                      peaks=self.query_spectrum['peaks'],
                                                      ms2_tolerance_in_da=0.02, target='gpu')
        np.testing.assert_almost_equal(similarity, [1.0, 0.22299, 0.66897, 0.66897], decimal=5)

    def test_neutral_loss_search(self):
        similarity = self.flash_entropy.neutral_loss_search(precursor_mz=self.query_spectrum['precursor_mz'],
                                                            peaks=self.query_spectrum['peaks'],
                                                            ms2_tolerance_in_da=0.02, target='gpu')
        np.testing.assert_almost_equal(similarity, [1.0, 0.0, 0.44598, 0.22299], decimal=5)

    def test_open_search(self):
        similarity = self.flash_entropy.open_search(peaks=self.query_spectrum['peaks'], ms2_tolerance_in_da=0.02, target='gpu')
        np.testing.assert_almost_equal(similarity, [1.0, 0.22299, 0.22299, 0.44598], decimal=5)

    def test_identity_search(self):
        similarity = self.flash_entropy.identity_search(precursor_mz=self.query_spectrum['precursor_mz'],
                                                        peaks=self.query_spectrum['peaks'],
                                                        ms1_tolerance_in_da=0.01, ms2_tolerance_in_da=0.02, target='gpu')
        np.testing.assert_almost_equal(similarity, [1.0, 0.0, 0.0, 0.0], decimal=5)


if __name__ == '__main__':
    unittest.main()
