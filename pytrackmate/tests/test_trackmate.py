import os
import xml.etree.cElementTree as et

import numpy as np
import pandas as pd

from pytrackmate import trackmate_peak_import, trackmate_peak_export


def test_import():
    fname = os.path.join(os.path.dirname(__file__), "FakeTracks.xml")
    spots = trackmate_peak_import(fname)
    assert spots.shape == (12, 17)


def test_export(tmp_path):
    test_dataframe = pd.DataFrame(
        {
            "t_stamp": {0: 0.0, 7: 0.0},
            "t": {0: 0.0, 7: 0.0},
            "x": {0: 28.81291282851087, 7: 19.267282188493684},
            "y": {0: 5.666830103137038, 7: 5.783638575972015},
            "z": {0: 0.0, 7: 0.0},
            "mean_intensity": {0: 50.22680412371134, 7: 51.22680412371134},
            "w": {0: 2.0000000141168894, 7: 2.0000000141168894},
            "q": {0: 7.669281005859375, 7: 8.05587100982666},
            "spot_id": {0: 35.0, 7: 10.0},
            "median_intensity": {0: 28.0, 7: 30.0},
            "min_intensity": {0: 0.0, 7: 2.0},
            "max_intensity": {0: 248.0, 7: 239.0},
            "total_intensity": {0: 4872.0, 7: 4969.0},
            "std_intensity": {0: 58.9290719768688, 7: 59.27381820040769},
            "contrast": {0: 0.2350598400953469, 7: 0.26399855726563876},
            "snr": {0: 0.32443401070344374, 7: 0.36101008108515836},
            "label": {0: 0, 7: 1},
        }
    )
    result_xml = trackmate_peak_export(test_dataframe)
    assert "Spot" in result_xml

    test_xml_file = tmp_path / "Tracks.xml"
    test_xml_file.write_text(result_xml)

    test_dataframe2 = trackmate_peak_import(test_xml_file)

    for k in test_dataframe2.keys():
        assert k in test_dataframe.keys()
        print(test_dataframe[k])
        print(test_dataframe2[k])
        assert np.all(np.isclose(test_dataframe[k], test_dataframe2[k]))
