# Algorithmisation and Implementation of Wason Selection Task Theories in Python

Review of four existing theories regarding the Wason Selection Task

* Quantitive Optimal Data Selection
* Inference-Guessing Model
* PSYCOP
* Mental Model

### Requirements

The multinomial process trees are analysed with the R framework MPTinR

* MPTinR for analysing multinomial processing trees (https://cran.r-project.org/web/packages/MPTinR/MPTinR.pdf)

For Quantitive Optimal Data Selection we use the following libraries

* numpy
* pandas
* matplotlib
* scipy

# Running the models

For the inference guessing model the 5 parameters must be determined c="CONDITIONAL", d="FORWARD", s="SUFFICIENT", i="IRREVERSIBLE", x="BIDIRECTIONAL".

For psycop the 2 parameters must be determined c="CONDITIONAL", i="imagine".

## Running the analysis

To analyze the multinomial process trees the script fit.r can be started.

To analyze QODS the quantitative_optimal_data_selection.py can be started

## Authors

* **Mathias Hirth**
* **Prof. Dr.** (Mental Model)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
