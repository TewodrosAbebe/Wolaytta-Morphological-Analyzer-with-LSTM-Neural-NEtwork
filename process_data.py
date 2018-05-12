import numpy as np
from keras.utils import np_utils

# n_sample = number of examples in the data
# n_chars = number of charactor - 26
# n_feature = number features for each example - len(features)
# n_max_root_chars = maximum number of characters in a root word
# n_max_target_chars = maximum number of characters in a target word


# create a dictionary that contains all word features and thier
# possible values
features = {
    "person": ['p1', 'p2', 'p3'],
    "number": ['sing', 'plr'],
    "gender": ["masc", 'fem'],
    "aspect": ['prf', 'imfr'],
    "verb_type": ['tv', 'iv'],
    "causative": ['cau'],
    "relative": ['rel'],
    "future": ['fut'],
    "optative": ['opt'],
    "interrogative_ending": ['intEnding'],
    "converb": ["shortcnvb", "longcnvb"],
    "simultaneous": ['sim']
}
# create a dictionary that contains index of the feature names
# and the feature names themselevs so that we can access the
# features by indexes
feature2int = dict((i, key) for i, key in enumerate(features.keys()))


def get_raw_data(filename):
    """ Read the file and return the list of rows on the file

        Example: [
                    ['b v iv p3 sing masc imfr ', 'bees']
                 ]

    Arguments:
    filename -- A file name of the file containing the word data

    Return:
    data -- list containing the features and outputs of word roots
    """
    data = []
    with open(filename, mode='r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split('\t')
            ps = []
            for part in parts:
                if len(part) > 0 and part is not '\n':
                    ps.append(part)
            data.append(ps)
        print(ps)
    return data


def setup_features_and_targets(data):
    """ Returns vectorized form of the word data

        The root and target words are represented with individual letters as one-hot
        encoding of the ascii number

        Other features are represented by one-hot encoding of the feature, having
        the feature index as a value

    Arguments:
    data -- a python list containing the features and the target of each word

    Return:
    x_roots -- a (n_sample X n_max_root_chars X n_chars) numpy array containg all
                one-hot encoding for each letters for each
                root word in the example list

    x_features -- a (n_sample X n_feature) numpy array features of each word in the sample

    x_targets -- a (n_sample X n_max_target_chars X n_chars) numpy array containg all
                one-hot encoding for each letters for each
                root word in the example list

    max_root_chars -- python integer of the n_max_root_chars

    max_target_chars --  python integer of the n_max_target_chars
    """
    x_features = []
    x_roots = []
    y_targets = []

    max_root_chars = 0
    max_target_chars = 0

    keys = features.keys()
    for d in data:
        fs = d[0]
        target = d[1]
        feat_split = fs.split(' ')

        root = [ord(c) - 97 for c in feat_split[0]]
        target = [ord(c) - 97 for c in target]
        x_roots.append(root)
        y_targets.append(target)

        ft = [-1] * len(features)

        for feat in feat_split[2:]:
            for i, f in enumerate(keys):
                if feat in features[f]:
                    index = features[f].index(feat)
                    ft[i] = index
                    break
        x_features.append(ft)

        if len(root) > max_root_chars:
            max_root_chars = len(root)
        if len(target) > max_target_chars:
            max_target_chars = len(target)

    return x_roots, x_features, y_targets, max_root_chars, max_target_chars


def to_one_hot(index, size):
    """ Returns a vector of size one-hot encoding of the index

    Arguments:
    index -- the index on which the vector becomes 1

    size --  the number of elements of the vector

    Return:
    x -- the one-hot vector
    """
    x = np.zeros((1, size))
    x[0, index] = 1
    return x


def one_hot_decode(mat):
    """ Returns a vector of decode of one-hot matrix

    Arguments:
    max -- a (n_sample X vec_size) matrix containing one-hot vectors

    Return:
    x -- a (n_sample X 1) vector containing decode of one-hot vector
    """
    x = [np.argmax(vec) for vec in mat]
    return x


def create_input_output(x_roots, x_features, y_targets, max_root, max_target):
    sample_size = len(x_roots)
    x_size = max_root + len(x_features[0])
    y_size = max_target
    X = -2 * np.ones((sample_size, x_size))
    Y = -2 * np.ones((sample_size, y_size))
    for i in range(sample_size):
        root, feats, target = x_roots[i], x_features[i], y_targets[i]
        root_hot = np_utils.to_categorical([root], num_classes=26)
        target_hot = np_utils.to_categorical([target], num_classes=26)

        X[i, :len(root)] = root
        X[i, max_root:] = feats
        Y[i, :len(target)] = target
    return X, Y


def setup_data_for_seq2seq(x_roots, x_features, y_targets, max_root, max_target):
    """ Returns matrixes for seq2seq architecture from root, feature and target vectors

        the input to the decoder contains reverse input of target padded by SOS and EOS

    Arguments:
    x_roots -- a (n_sample X n_max_root_chars X n_chars) numpy array containg all
                one-hot encoding for each letters for each
                root word in the example list

    x_features -- a (n_sample X n_feature) numpy array features of each word in the sample

    y_targets -- a (n_sample X n_max_target_chars X n_chars) numpy array containg all
                one-hot encoding for each letters for each
                root word in the example list

    max_root_chars -- python integer of the n_max_root_chars

    max_target_chars --  python integer of the n_max_target_chars

    Return:
    X_encoder -- a (n_sample X n_max_root_chars X n_chars) tensor representing the root words
    X_decoder -- a (n_sample X n_max_target_chars X n_chars + 2) tensor representing the
                target words input to the decoder
    Y -- a (n_sample X n_max_target_chars X n_chars + 2) tensor representing the
                target words output to the decoder
    """
    vocab_size = 26 + len(x_features[0])
    SOS = ord('{') - 97
    EOS = ord('|') - 97
    vocab_size_decoder = vocab_size + 2
    sample_size = len(x_roots)
    X_encoder = np.ones((sample_size, max_root, vocab_size))
    X_decoder = np.ones((sample_size, max_target, vocab_size_decoder))
    Y = np.ones((sample_size, max_target, vocab_size_decoder))
    for i in range(1):
        root = x_roots[i]
        feat = x_features[i]
        target = y_targets[i]

        d_target = target[:-1]
        d_target.reverse()
        decoder_input = [SOS] + d_target + [EOS]
        target = [SOS] + target + [EOS]

        hot_root = np_utils.to_categorical(root, num_classes=vocab_size)

        hot_decoder_input = np_utils.to_categorical(
            decoder_input, num_classes=vocab_size_decoder)
        hot_target = np_utils.to_categorical(
            target, num_classes=vocab_size_decoder)

        X_encoder[i, :len(root), :] = hot_root
        X_decoder[i, :len(decoder_input), :] = hot_decoder_input
        Y[i, :len(target), :] = hot_target

    return X_encoder, X_decoder, Y
