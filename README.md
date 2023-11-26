
## Structure
- `Additional Data/` contains the peptide data split for cross validation
- `csvs/` contains the model configurations required to reproduce classical and quantum results in the paper; additional parameters can be added by examining the `arg_type_map` variable in `run.py`
- `model/` contains the TorchQuantum and PyTorch quantum and classical models, as well as utility files
- `utils/` contains other miscellaneous functions used in processing

- `run.py` and `run_classical.py` run the specified model and save data using Tensorboard
- `average_folds.py` averages test and validation F1 data from executed runs
- `get_attributions.py` and `get_attributions_classical.py` compute the Captum IG/SVS attributions

## Modifications to TorchQuantum
A small number of modifications have to be made to TorchQuantum to support the recurrent methods used here. As of TQ 0.1.7, these are:
1. Replace line 90 in the `GeneralEncoder` class in `encoding/encodings.py` with the following four lines:
    ```python
    50  class GeneralEncoder(Encoder, metaclass=ABCMeta):

        ...

    90      def forward(self, qdev: tq.QuantumDevice, x, reset=True):
    91          self.q_device = qdev
    92          if reset:
    93             self.q_device.reset_states(x.shape[0])
    ```
i.e. add `reset` as a default parameter and add the `if` statement resetting the state.

2. Create the `MeasureOne` class in `measurement/measurements.py`:
    ```python
    class MeasureOne(tq.QuantumModule):
        """Obtain the expectation value of all the qubits."""

        def __init__(self, obs, v_c_reg_mapping=None):
            super().__init__()
            self.obs = obs
            self.v_c_reg_mapping = v_c_reg_mapping

        def forward(self, qdev: tq.QuantumDevice):
            x = expval(qdev, [0], [self.obs()])

            if self.v_c_reg_mapping is not None:
                c2v_mapping = self.v_c_reg_mapping["c2v"]
                """
                the measurement is not normal order, need permutation
                """
                perm = []
                for k in range(x.shape[-1]):
                    if k in c2v_mapping.keys():
                        perm.append(c2v_mapping[k])
                x = x[:, perm]

            if self.noise_model_tq is not None and self.noise_model_tq.is_add_noise:
                return self.noise_model_tq.apply_readout_error(x)
            else:
                return x

        def set_v_c_reg_mapping(self, mapping):
            self.v_c_reg_mapping = mapping
    ```
This is identical to the pre-existing `MeasureAll` class except for the first line of the `forward` method, which here returns only a single observable. 

Then add `MeasureAll` to the `__all__` list defined on line 17 in the same file.

## Usage
1. Install the requirements in a `venv` by using `pip install -r requirements.txt`.
2. Then install [TorchQuantum](https://github.com/mit-han-lab/torchquantum) in editable mode following the instructions on their homepage, and the modifications required above.
3. Create a `.csv` file detailing the structure of the model you wish to run (`csvs/` contain some examples of this).
4. Attributions can be calculated and the best F1 score over folds found by using `get_attributions.py` and `average_folds.py`.