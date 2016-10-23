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
        for filename in filenames:
            recorder.creat_example(extract_features(filename))


if __name__ == '__main__':
    import sys

    folder = sys.argv[1]
    glob_str = sys.argv[2]

    main(glob.glob(folder + glob_str))
