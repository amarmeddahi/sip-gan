# SIP-GAN
This package provides an implementation of the SIP-GAN generative method. This is a new model for SIP traffic generation that was published in ISNCC-2021. For simplicity, we refer to this model as SIP-GAN throughout the rest of this document.

Any publication that discloses findings arising from using this source code or
the model parameters should [cite](#citing-this-work) the SIP-GAN paper.

## First time setup

The following steps are required in order to run SIP-GAN:

1.  Install [Spyder](www.spyder-ide.org) (we recommend the cross-platform [Anaconda](www.anaconda.com) distribution).
    *   Install
[TensorFlow](www.tensorflow.org/install). 
    *   For GPU support: [TensorFlow GPU](www.tensorflow.org/install/gpu).
1.  Check that SIP-GAN will be able to use a GPU by running:

```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```
The output of this command should show a list of your GPUs. If it doesn't,
check if you followed all steps correctly when setting up [TensorFlow GPU](www.codingforentrepreneurs.com/blog/install-tensorflow-gpu-windows-cuda-cudnn).

## Running SIP-GAN

1. Clone this repository and `cd` into `/code`.

    ```bash
    git clone https://github.com/amarmeddahi/sip-gan
    ```
1. Data preprocessing + encoder + decoder
* text file --> Matrix: Run in `code/dataset.py` the `Data Preprocessing` part.
* Matrix --> Images : Run in `code/dataset.py` the `Encoder` part.
* Images --> text file: Run in `code/dataset.py` the `Decoder` part.
3. Generator
* Run `code/generator.py` (you can modify the parameters: batch size, epsilon, maximum iteration…).  
	* Use the decoder part in `code/dataset.py` to create a text file from the generated samples ([txt2pcap](www.wireshark.org/docs/man-pages/text2pcap.html) to convert the text file into a pcap file that can be analyzed with a network packet analyzer).
## Citing this work

If you use the code or data in this package, please cite:

```bibtex
@Article{SIPGAN2021,
  author  = {Amar Meddahi and Hassen Drira and Ahmed Meddahi},
  journal = {2021 International Symposium on Networks, Computers and Communications
(ISNCC): Artificial Intelligence and Machine Learning (ISNCC-2021 AIML)
},
  title   = {{SIP-GAN:} Generative Adversarial Networks for {SIP} traffic generation},
  year    = {2021},
}
```

## Acknowledgements

SIP-GAN communicates with and/or references the following separate libraries
and packages:

*   [matplotlib](https://matplotlib.org/)
*   [NumPy](https://numpy.org)
*   [TensorFlow](https://github.com/tensorflow/tensorflow)

We thank all their contributors and maintainers!
