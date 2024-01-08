# Local Environment of Fluorine (LEF) Shift Prediction Tools

## Getting Started

A PyPi package is available:
```
pip install lefshift
```

You can also clone the repo manually and then install the lefshift package:
```
git clone https://github.com/PatrickPenner/lefshift.git
cd lefshift
pip install .
```
Be aware that the `data/` and `model/` directory used in the following are in
the `lefshift` repo.

Running `lefshift --help` should give you an overview of how to use the lefshift
commandline tool. You can quickly train a model using the Enamine data in `data/`
like this:
```
lefshift train data/train.csv --model models/my_model --id-column 'Catalog ID' --shift-column 'Shift 1 (ppm)' --smiles-column 'SMILES' --verbose
```
Training requires 3 columns: an ID column, a chemical shift column, and a
SMILES column. The above command gives the name of those columns explicitly.
You can also check the `lefshift train --help` output to see the default column
names that lefshift expects.

You can perform a prediction with the model you just trained with the following command:
```
lefshift predict data/test.csv --model models/my_model data/test_predicted.csv --smiles-column 'SMILES' --verbose
```
Prediction only requires a SMILES column. This is once again give explicitly,
but you can also ensure that there is an input column named "SMILES".

To split a data set into those samples in the applicability domain of the model
and those not in the applicability domain run this command:
```
lefshift split data/test.csv --model models/my_model data/test_known.csv data/test_unknown.csv --verbose
```
Splitting also only requires a SMILES column. The SMILES column in the input is
already called "SMILES" so we don't have to be explicit.

For more advanced usage please check the `--help` output of `lefshift` and the
subtools `train`, `predict`, and `split`. There is also a
`QM Assisted ML Tutorial.md` that describes a workflow to combine `lefshift`
and `lefqm`.

## Development

### Formatting pre-commit hook

Pre-commit is only used for consistent formatting. The core of that is the
black formatter and code style.

Install the formatting pre-commit hook with:
```
pre-commit install
```

### Code Quality

All quality criteria have a range of leniency.

| Criteria               | Threshold     |
| -------------          |:-------------:|
| pylint                 | \>9.0         |
| coverage (overall)     | \>90%         |
| coverage (single file) | \>80%         |

### Utility commands

Pre-commit on one file:
```
pre-commit run --files
```

Test command:

```
python -m unittest tests
```

PyLint command:

```
pylint lefshift tests > pylint.out
```

Coverage commands:
```
coverage run -m unittest tests
coverage html
```
