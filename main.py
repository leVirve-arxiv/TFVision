import glob
import scipy.misc

from tfrecord import Recorder, Feature


def extract_features(filename):
    image = scipy.misc.imread(filename)
    raw = image.tostring()
    rows, cols, channels = image.shape

    return {
        'height': Feature(int64_list=cols),
        'width': Feature(int64_list=rows),
        'channel': Feature(int64_list=channels),
        'image_raw': Feature(bytes_list=raw)
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
