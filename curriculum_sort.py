import argparse
import os
import random


def main():
    args = parse_args()
    dataset = load_dataset(args.source_data, args.target_data)
    vocabulary = load_vocabulary(args.vocabulary)
    bins = assign_to_bins(dataset, vocabulary, args.threshold, args.side, args.averaged)

    write_to_files(bins, args.out_dir)
    print_stats(bins)
   

def parse_args():
    parser = argparse.ArgumentParser(description='Create a curriculum for a parallel data set')
    parser.add_argument("source_data", help="Path to data set: source language")
    parser.add_argument("target_data", help="Path to data set: target language")
    parser.add_argument("vocabulary", help="Path to vocabulary (needs to inculde word frequencies)")
    parser.add_argument("out_dir", help="Directory for output files")
    parser.add_argument("-threshold", type=int, action="append", 
        help="The threshold (min/average word freq.) for each curriculum level (each bin)")
    parser.add_argument("-side", default="target", help="Side of corpus to base sorting on: source/target")
    parser.add_argument("-averaged", action='store_true', help="Base ordering on average word frequencies")
    args = parser.parse_args()

    # make sure at least one threshold is given
    if args.threshold == None:
        raise ValueError("No thresholds given. Provide at least one threshold for sorting")
    else:
        return args


def load_dataset(source_path, target_path):
    """Read parallel data from file (includes tokenization)

    Arguments:
            source_path {str} -- path to source language text file
            target_path {str} -- path to target language text file

    Returns:
            list -- list containing tuples of sentence pairs (tokenized)
    """
    with open(source_path) as source_file:
        lines_source = [line.split() for line in source_file.readlines()]

    with open(target_path) as target_file:
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
    with open(path) as vocab_file:
        lines = vocab_file.readlines()[1:]  # skip column headers

        for line in lines:
            splitted = line.split("\t")
            vocab_dict[splitted[0]] = int(
                splitted[1].strip("\n"))

    return vocab_dict


def assign_to_bins(dataset, vocabulary, thresholds, side, averaged):
    """Assign data to bins of different difficulty

    A sentence qualifies for a specific bin, if it's rank  
    exceeds the according threshold

    Example:
    bins[100] contains all sentences with rank > 100,
    bins[10] contains all sentences with 100 > rank > 10

    Arguments:
            dataset {list} -- list of tuples of tokenized sent-pairs
            vocabulary {dict} -- dict with word:freq
            thresholds {list} -- the values needed to qualify a sent for an according bin
            side {str} -- "source" or "target": side of corpus to base sorting on  
            averaged{bool} -- whether to use averaged word frequencies

    Returns:
            dict -- key: bin-threshold, val: list containing all qualified sent-pairs
    """
    bins = {t: [] for t in thresholds}

    if side == "source":
        idx = 0
    elif side == "target":
        idx = 1

    for sent_pair in dataset:
        sent_rank = rank_sentence(sent_pair[idx], vocabulary, averaged)

        for threshold in sorted(bins.keys(), reverse=True):
            if sent_rank >= threshold:
                bins[threshold].append(sent_pair)
                break

    # remove empty bins
    keys = list(bins.keys())
    for key in keys:
        if bins[key] == []:
            print("Bin with threshold {} is empty.".format(key))
            del bins[key]

    return bins


def rank_sentence(sentence, vocabulary, averaged):
    """Determine rank of a sentence

    A rank is equal to either:
    - the lowest occuring word frequency in the sentence (averaged==False)
    - the average frequency of the words in the sentence (averaged==True)

    Arguments:
            sentence {str} -- the sentence to rank
            vocabulary {dict} -- word:freq
            average {bool} -- whether to use averaged word frequencies

    Returns:
            number -- the rank of a sentence
    """
    frequencies = []

    for word in sentence:
        try:
            frequencies.append(vocabulary[word])
        except KeyError:
            frequencies.append(0)

    if averaged:
        summed_freqs = sum(frequencies)
        return int(summed_freqs/len(sentence))
    else:
        return min(frequencies)


def write_to_files(bins, out_dir):
    """Writes sorted data to files

    Each bin is written to a seperate file

    Arguments:
            bins {dict} -- threshold:[sent-pairs]
            out_dir {str} -- path to output directory
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for key in bins.keys():
        outfile_source = open("{}/src_{}.txt".format(out_dir, key), "w")
        outfile_target = open("{}/trg_{}.txt".format(out_dir, key), "w")

        for sent in bins[key]:
            outfile_source.write(" ".join(sent[0]))
            outfile_source.write("\n")

            outfile_target.write(" ".join(sent[1]))
            outfile_target.write("\n")

        outfile_source.close()
        outfile_target.close()

    print("\nFiles written to: {}".format(out_dir))


def print_stats(bins):
    """Prints several stats and sample sentences for each bin"""
    bins_keys = sorted(bins.keys(), reverse=True)
    print("\nNumber of curriculum levels (bins) created: {}".format(len(bins_keys)))
    print("Thresholds (min/average word frequencies): {}".format(bins_keys))
    print("Data points per bin: {}\n".format([len(bins[b]) for b in bins]))

    for key in bins_keys:
        current_bin = bins[key]
        random_idx = random.randint(0, len(current_bin))
        sample_sent = " ".join(current_bin[random_idx][1])
        print("Example bin (threshold {}):\n{}\n".format(key, sample_sent))


if __name__ == '__main__':
    main()
