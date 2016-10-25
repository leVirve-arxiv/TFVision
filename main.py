import glob
import scipy.misc

from tfrecord import Recorder, FeatureInt64, FeatureBytes


def extract_features(filename):
    image = scipy.misc.imread(filename)
    raw = image.tostring()
    rows, cols, channels = image.shape
    label = None

    return {
        'height': FeatureInt64(cols),
        'width': FeatureInt64(rows),
        'channel': FeatureInt64(channels),
        'label': FeatureInt64(label),
        'image_raw': FeatureBytes(raw)
    }


def main(filenames):
    with Recorder('output.tfrecords') as recorder:
        recorder.feature_extractor = extract_features
        for filename in filenames:
            recorder.create_example(filename)


if __name__ == '__main__':
    import sys

    folder = sys.argv[1]
    glob_str = sys.argv[2]

    main(glob.glob(folder + glob_str))
