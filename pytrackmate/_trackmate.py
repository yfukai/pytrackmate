import xml.etree.cElementTree as et
from xml.dom import minidom

import numpy as np
import pandas as pd

OBJECT_LABELS = {
    "FRAME": "t_stamp",
    "POSITION_T": "t",
    "POSITION_X": "x",
    "POSITION_Y": "y",
    "POSITION_Z": "z",
    "MEAN_INTENSITY": "I",
    "ESTIMATED_DIAMETER": "w",
    "QUALITY": "q",
    "ID": "spot_id",
    "MEAN_INTENSITY": "mean_intensity",
    "MEDIAN_INTENSITY": "median_intensity",
    "MIN_INTENSITY": "min_intensity",
    "MAX_INTENSITY": "max_intensity",
    "TOTAL_INTENSITY": "total_intensity",
    "STANDARD_DEVIATION": "std_intensity",
    "CONTRAST": "contrast",
    "SNR": "snr",
}

OBJECT_LABELS_INV = {v: k for k, v in OBJECT_LABELS.items()}

FEATURE_PROPERTIES = {
    "QUALITY": dict(name="Quality", shortname="Quality", dimension="QUALITY", isint="false"),
    "POSITION_X": dict(name="X", shortname="X", dimension="POSITION", isint="false"),
    "POSITION_Y": dict(name="Y", shortname="Y", dimension="POSITION", isint="false"),
    "POSITION_Z": dict(name="Z", shortname="Z", dimension="POSITION", isint="false"),
    "POSITION_T": dict(name="T", shortname="T", dimension="TIME", isint="false"),
    "FRAME": dict(name="Frame", shortname="Frame", dimension="NONE", isint="true"),
    "RADIUS": dict(name="Radius", shortname="R", dimension="LENGTH", isint="false"),
    "VISIBILITY": dict(name="Visibility", shortname="Visibility", dimension="NONE", isint="true"),
    "MANUAL_COLOR": dict(name="Manual spot color", shortname="Spot color", dimension="NONE", isint="true"),
    "MEAN_INTENSITY": dict(name="Mean intensity", shortname="Mean", dimension="INTENSITY", isint="false"),
    "MEDIAN_INTENSITY": dict(name="Median intensity", shortname="Median", dimension="INTENSITY", isint="false"),
    "MIN_INTENSITY": dict(name="Minimal intensity", shortname="Min", dimension="INTENSITY", isint="false"),
    "MAX_INTENSITY": dict(name="Maximal intensity", shortname="Max", dimension="INTENSITY", isint="false"),
    "TOTAL_INTENSITY": dict(name="Total intensity", shortname="Total int.", dimension="INTENSITY", isint="false"),
    "STANDARD_DEVIATION": dict(name="Standard deviation", shortname="Stdev.", dimension="INTENSITY", isint="false"),
    "ESTIMATED_DIAMETER": dict(name="Estimated diameter", shortname="Diam.", dimension="LENGTH", isint="false"),
    "CONTRAST": dict(name="Contrast", shortname="Constrast", dimension="NONE", isint="false"),
    "SNR": dict(name="Signal/Noise ratio", shortname="SNR", dimension="NONE", isint="false"),
}


def trackmate_peak_import(trackmate_xml_path, get_tracks=False):
    """Import detected peaks with TrackMate Fiji plugin.

    Parameters
    ----------
    trackmate_xml_path : str
        TrackMate XML file path.
    get_tracks : boolean
        Add tracks to label
    """

    root = et.fromstring(open(trackmate_xml_path).read())

    objects = []

    features = root.find('Model').find('FeatureDeclarations').find('SpotFeatures')
    features = [c.get('feature') for c in features.getchildren()] + ['ID']

    spots = root.find('Model').find('AllSpots')
    trajs = pd.DataFrame([])
    objects = []
    for frame in spots.findall('SpotsInFrame'):
        for spot in frame.findall('Spot'):
            single_object = []
            for label in features:
                single_object.append(spot.get(label))
            objects.append(single_object)

    trajs = pd.DataFrame(objects, columns=features)
    trajs = trajs.astype(np.float)

    # Apply initial filtering
    initial_filter = root.find("Settings").find("InitialSpotFilter")

    trajs = filter_spots(trajs,
                         name=initial_filter.get('feature'),
                         value=float(initial_filter.get('value')),
                         isabove=True if initial_filter.get('isabove') == 'true' else False)

    # Apply filters
    spot_filters = root.find("Settings").find("SpotFilterCollection")

    for spot_filter in spot_filters.findall('Filter'):

        trajs = filter_spots(trajs,
                             name=spot_filter.get('feature'),
                             value=float(spot_filter.get('value')),
                             isabove=True if spot_filter.get('isabove') == 'true' else False)

    trajs = trajs.loc[:, OBJECT_LABELS.keys()]
    trajs.columns = [OBJECT_LABELS[k] for k in OBJECT_LABELS.keys()]
    trajs['label'] = np.arange(trajs.shape[0])

    # Get tracks
    if get_tracks:
        filtered_track_ids = [int(track.get('TRACK_ID')) for track in root.find('Model').find('FilteredTracks').findall('TrackID')]

        label_id = 0
        trajs['label'] = np.nan

        tracks = root.find('Model').find('AllTracks')
        for track in tracks.findall('Track'):

            track_id = int(track.get("TRACK_ID"))
            if track_id in filtered_track_ids:

                spot_ids = [(edge.get('SPOT_SOURCE_ID'), edge.get('SPOT_TARGET_ID'), edge.get('EDGE_TIME')) for edge in track.findall('Edge')]
                spot_ids = np.array(spot_ids).astype('float')[:, :2]
                spot_ids = set(spot_ids.flatten())

                trajs.loc[trajs["spot_id"].isin(spot_ids), "label"] = label_id
                label_id += 1

        # Label remaining columns
        single_track = trajs.loc[trajs["label"].isnull()]
        trajs.loc[trajs["label"].isnull(), "label"] = label_id + np.arange(0, len(single_track))

    return trajs


def filter_spots(spots, name, value, isabove):
    if isabove:
        spots = spots[spots[name] > value]
    else:
        spots = spots[spots[name] < value]

    return spots


def trackmate_peak_export(
    trackmate_dataframe,
    version="3.7.0",
    spatialunits="pixel",
    timeunits="sec",
    image_data_props={},
):
    """Export peaks to XML for TrackMate Fiji plugin.

    Parameters
    ----------
    trackmate_dataframe : pandas.DataFrame
        Trackmate peaks in form of pandas DataFrame

    Returns
    -------
    trackmate_xml : str
        Trackmate XML as string

    """

    root = et.Element("TrackMate", version=version)
    model = et.SubElement(root, "Model", spatialunits=spatialunits, timeunits=timeunits)
    feature_declarations = et.SubElement(model, "FeatureDeclarations")
    spot_features = et.SubElement(feature_declarations, "SpotFeatures")
    for key in trackmate_dataframe.keys():
        if key in OBJECT_LABELS_INV.keys() and key != "spot_id":
            original_key = OBJECT_LABELS_INV[key]
            properties = FEATURE_PROPERTIES[original_key]
            et.SubElement(spot_features, "Feature", feature=original_key, **properties)

    all_spots = et.SubElement(model, "AllSpots", nspots=str(len(trackmate_dataframe)))
    for t_stamp, grp in trackmate_dataframe.groupby("t_stamp"):
        spots_in_frame = et.SubElement(all_spots, "SpotsInFrame", frame=str(t_stamp))
        for _, row in grp.iterrows():
            row_dict = row.to_dict()
            del row_dict["spot_id"]
            to_string = (
                lambda k, v: str(int(v))
                if FEATURE_PROPERTIES[OBJECT_LABELS_INV[k]]["isint"] == "true"
                else str(float(v))
            )
            row_dict = {
                OBJECT_LABELS_INV[k]: to_string(k, v)
                for k, v in row_dict.items()
                if k in OBJECT_LABELS_INV
            }
            row_dict["ID"] = str(int(row["spot_id"]))
            et.SubElement(spots_in_frame, "Spot", **row_dict)

    settings = et.SubElement(root, "Settings")
    et.SubElement(settings, "ImageData")
    et.SubElement(settings, "BasicSettings")
    et.SubElement(
        settings, "InitialSpotFilter", feature="QUALITY", value="0.0", isabove="true"
    )
    et.SubElement(settings, "SpotFilterCollection")

    trackmate_xml = et.tostring(root)
    trackmate_xml = minidom.parseString(trackmate_xml)
    return trackmate_xml.toprettyxml(indent="  ")
