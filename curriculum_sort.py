import argparse
import os


def main():
    args = parse_args()
    dataset = load_dataset(args.source_data, args.target_data)
    vocabulary = load_vocabulary(args.vocabulary)

    if args.threshold != None:
        bins = assign_to_bins(dataset, vocabulary,
                              args.num_bins, args.threshold, args.side)
    # else:
    # 	bins = auto_assign_to_bins(dataset, vocabulary, args.num_bins, args.s)

    write_to_files(bins, args.out_dir)

    #--Stats--
   

def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a Curriculum for parallel data')
    parser.add_argument("source_data", help="source data")
    parser.add_argument("target_data", help="target data")
    parser.add_argument("vocabulary", help="vocabulary")
    parser.add_argument("out_dir", help="directory for output")
    parser.add_argument("num_bins", type=int,
                        help="number of curriculum levels/stages")
    parser.add_argument("-threshold", type=int, action="append", help="the thresholds")
    parser.add_argument("-side", default="target",
                        help="side of corpus: source/target")
    return parser.parse_args()


def load_dataset(source_path, target_path):
    """Read parallel data from file

    Arguments:
            source_path {str} -- path to source language text file
            target_path {str} -- path to target language text file

    Returns:
            list -- list containing tuples of sentence pairs (tokenized)
    """
    source_file = open(source_path)
    target_file = open(target_path)

    lines_source = [line.split() for line in source_file.readlines()]
    lines_target = [line.split() for line in target_file.readlines()]

    return list(zip(lines_source, lines_target))


def load_vocabulary(path):
    """Read vocabulary from file

    Vocab file must be of format: 
    word tab frequency

    Arguments:
            path {str} -- path to vocabulary file

    Returns:
            dict -- key: word, value: frequency
    """
    vocab_dict = dict()
    with open(path, encoding="utf-8") as vocab_file:
        lines = vocab_file.readlines()[1:]  # skip column headers

        for line in lines:
            splitted = line.split("\t")
            vocab_dict[splitted[0]] = int(
                splitted[1].strip("\n"))

    return vocab_dict


def assign_to_bins(dataset, vocabulary, num_bins, thresholds, side):
    """Assign data to bins of different difficulty

    A sentence qualifies for a specific bin, if it's rank  
    exceeds the according threshold

    example:
    bin[100] contains all sentences with rank > 100,
    bin[10] contains all sents with 100 > rank > 10

    Arguments:
            dataset {list} -- list of tuples of tokenized sent-pairs
            vocabulary {dict} -- dict with word:freq
            num_bins {int} -- number of desired bins
            thresholds {list} -- the values needed to qualify a sent for an according bin
            side {str} -- "source" or "target": side of corpus to base sorting on  

    Returns:
            dict -- key: bin-threshold, val: list containing all qualified sent-pairs
    """
    bins = {t: [] for t in thresholds}
    bins[0] = []

    if side == "source":
        idx = 0
    elif side == "target":
        idx = 1

    for sent_pair in dataset:
        sent_rank = rank_sentence(sent_pair[idx], vocabulary)

        for threshold in sorted(bins.keys(), reverse=True):
            if sent_rank >= threshold:
                bins[threshold].append(sent_pair)
                break
    return bins


# def auto_assign_to_bins():


def rank_sentence(sentence, vocabulary):
    """Determine rank of a sentence

    A rank is equal to the lowest occuring word
    frequency in the sentence

    Arguments:
            sentence {str} -- the sentence to rank
            vocabulary {dict} -- word:freq

    Returns:
            number -- the rank (aka. lowest frequency)
    """
    lowest_frequency = max(vocabulary.values())

    for word in sentence:
        try:
            word_freq = vocabulary[word]
        except KeyError:
            return 0

        if word_freq < lowest_frequency:
            lowest_frequency = word_freq

    return lowest_frequency


def write_to_files(bins, out_dir):
    """Writes sorted data to files

    Each bin is written to a seperate file

    Arguments:
            bins {dict} -- threshold:[sent-pairs]
            out_dir {str} -- path to output directory
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print("Writing files to: {} ...".format(out_dir))

    for key in bins.keys():
        outfile_source = open("{}/src_{}.txt".format(out_dir, key), "w")
        outfile_target = open("{}/trg_{}.txt".format(out_dir, key), "w")

        for sent in bins[key]:
            outfile_source.write(" ".join(sent[0]))
            outfile_source.write("\n")

            outfile_target.write(" ".join(sent[1]))
            outfile_target.write("\n")

    print("... Done!")


if __name__ == '__main__':
    main()
