### motion_pred_state_block

* motion_seq2seq_model

  To incorporate this model, use the following command:
  ```bash
  python src/translate.py --action walking --seq_length_out 25 --omit_one_hot True --architecture basic --loss_to_use supervised
  ```
### Dependencies
* [h5py](http://www.h5py.org/)
* [tensorflow](https://www.tensorflow.org/) 1.2 or later

### Acknowledgments
This code is adapted from [human-motion-prediction](https://github.com/rajeev595/human-motion-prediction) repository by [Julieta Martinez](https://github.com/una-dinosauria/)
