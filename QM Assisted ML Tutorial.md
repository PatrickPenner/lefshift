# QM Assisted ML Tutorial

## Dependencies

Ensure that `lefshift`, `lefqm`, as well as all tools that `lefqm` needs are
installed. Please see the README of `lefqm` for tool workarounds if not all of
them are available.

The tutorial assumes you are working in the `lefshift` repo project directory.

## ML Prediction

Train a model based on Enamine data:
```
lefshift train data/train.csv --model models/model --id-column 'Catalog ID' --shift-column 'Shift 1 (ppm)' --verbose
```

Explanations for all commandline arguments can be found in top-level our tool
level help:
```
lefshift --help  # top-level help that shows what tools are available
lefshift train --help  # tool-level help for all commandline arguments
```

Perform a simple prediction of the test set with the following command:
```
lefshift predict data/test.csv --model models/model data/test_predicted.csv --similarities data/test_similarities.csv --verbose
```
This will also output the most similar training examples for each input
molecule. Looking at the output in `test_predicted.csv` we can see that the
molecule "Z1481144763" is not well predicted. Its closest analog in the
training set is not very similar. Let us split off those molecules that are
not covered by the training set of our model:
```
lefshift split data/test.csv --model models/model data/known.csv data/unknown.csv --verbose
```

## QM Prediction

Having split off the molecules that are not well-predicted by ML, we can
process them by QM. We start by generating conformations for these molecules:
```
mkdir -p unknown_conformers/
lefqm conformers data/unknown.csv unknown_conformers/ --id-column 'Catalog ID'
```
Running this job on your own machine may be fast enough for one molecule, but
for more you should probably use a cluster. Conformers are written into
`unknown_conformers/` as SD files, named after one input molecule, that
contain all conformers for that molecule.

Shieldings for conformer ensembles are calculated like this:
```
mkdir -p unknown_shieldings/
lefqm shieldings unknown_conformers/Z1481144763.sdf unknown_shieldings/Z1481144763.sdf
```
Here too, running this on you own machine may be fast enough for one ensemble,
but for more you should probably use a cluster.

Shieldings for an ensemble can be combined using:
```
lefqm ensembles unknown_shieldings/Z1481144763.sdf Z1481144763.csv
```
... or ...
```
lefqm ensembles unknown_shieldings/ unknown_shieldings.csv
```
... to combine all ensembles in a directory into one shieldings CSV.

Shieldings can be converted to shift with a calibration data set in the
following way:
```
sed -i 's/ID,/Catalog ID,/' unknown_shieldings.csv  # rename the ID column to make it consistent with the input data
lefqm shifts unknown_shieldings.csv --calibration data/train.csv unknown_shifts.csv --id-column 'Catalog ID' --shift-column 'Shift 1 (ppm)'
```
The calibration data set must contain a shieldings constants column as well
as a chemical shift column to train the linear regression conversion from
shieldings constants to chemical shifts. The file `unknown_shfits.csv` now
contains QM-derived chemical shifts for as many of the molecules from
`unknown.csv` as could be processed.

## QM Assisted ML Prediction

We can now go back into ML with the QM-derived chemical shifts. We will start
by combining `unknown.csv` and `unknown_shifts.csv` together in a format that
mirrors our training data. On the commandline this can look like this:
```
cut -d',' -f1 unknown_shifts.csv | while read line; do grep "$line" data/unknown.csv; done | cut -d':' -f 2 > unknown_processed.csv  # get all the input for molecules that could be processed
paste -d',' <(cut -d',' -f-10 unknown_processed.csv) <(cut -d',' -f5 unknown_shifts.csv) <(cut -d',' -f12- unknown_processed.csv) <(cut -d',' -f4 unknown_shifts.csv)> unknown_training.csv  # replace true chemical shift column in input with QM-derived shifts
```
It is probably simpler to do this in the visual spreadsheet editor of your
choice. The commands above also appended the shieldings column for consistency,
but we will not be using it for ML training.

Now we can combine `unknown_training.csv` with the `train.csv`:
```
cp data/train.csv qm_assisted_train_shieldings.csv
grep -v 'Catalog ID' unknown_training.csv >> qm_assisted_train_shieldings.csv  # do not add the header again
```

The file `qm_assisted_train_shieldings.csv` can now be used to train a model
with QM-derived data and repredict our original input:
```
lefshift train qm_assisted_train_shieldings.csv --model models/qm_assisted_model --id-column 'Catalog ID' --shift-column 'Shift 1 (ppm)' --verbose
lefshift predict data/test.csv --model models/qm_assisted_model data/test_qm_assisted_predicted.csv --similarities data/test_qm_assisted_similarities.csv --verbose
```
The prediction for a few molecules, such as "Z1481144763", has improved.
