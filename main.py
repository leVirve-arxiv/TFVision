import glob
import scipy.misc

from tfrecord import Recorder, Example, Feature


def creat_example(filename):
    image = scipy.misc.imread(filename)
    raw = image.tostring()
    rows, cols, channels = image.shape

    features = {
        'height': Feature(int64_list=cols),
        'width': Feature(int64_list=rows),
        'channel': Feature(int64_list=channels),
        'image_raw': Feature(bytes_list=raw)
    }

    return Example(features)


def main(filenames):
    with Recorder('output.tfrecords') as recorder:
        for filename in filenames[:100]:
            recorder.write(creat_example(filename))


if __name__ == '__main__':
    import sys

    folder = sys.argv[1]
    glob_str = sys.argv[2]

    main(glob.glob(folder + glob_str))
