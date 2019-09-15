# curriculum_learning

This repository includes a script for creating an ordered curriculum for a parallel data set (e.g. for a machine translation task).

More information on curriculum learning:
- [Curriculum Learning - Bengio et al., 2009](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf)
- [Curriculum Learning for NMT - Kocmi/Bojar, 2017](https://www.acl-bg.org/proceedings/2017/RANLP%202017/pdf/RANLP050.pdf)

## Run
Run ```python curriculum_sort.py -h``` for options

## Example
Running the following will create three separate files - one for each level of the curriculum:

    $ python curriculum_sort.py data/test_data.fr data/test_data.en data/vocab.en ./output 3 -threshold 5000 -threshold 100 -threshold 10
    
The thresholds correspond to the three bins and deterimine what the minimum frequency of a word in a sentence needs to be, to qualify the entire sentence pair for a bin.
