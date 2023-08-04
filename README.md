## ProactiV: Studying deep learning model behavior under input transformations

Deep learning (DL) models have shown performance benefits across many applications, from classification to, more recently, image-to-image translation. However, low interpretability often leads to unexpected model behavior when deployed in the real world. Usually, this unexpected behavior is because the training data domain does not reflect the deployment data domain. Identifying model breaking points under input conditions and domain shifts, i.e., input transformations, is essential to improve models.

ProactiV is a DL model-agnostic visual analytics method to help model developers proactively study output behavior under input transformations to identify and verify breakpoints. It relies on a proposed input optimization method to determine the changes required to a transformed input to achieve the desired output. The data from this optimization process allows the study of global and local model output behavior under input transformations at scale. Additionally, the optimization method provides insight into the input characteristics that result in desired outputs and helps recognize model biases.

## Input Optimization

![Overview of the proposed input optimization method](https://github.com/vidyaprsd/proactiv/blob/dev/imgs/overview.png)

## Repository
<b>This repository contains scripts for the proposed input optimization process of ProactiV</b>. While ProactiV is generic to any differentiable model, we provide a character classification example with EMNIST in this repository. More complex image-to-image translation use cases including MRI reconstruction and GANs can be found in our paper.

## Execution
The scripts mostly run independently from each other. 

- First, set the parameters for the input optimization, logging, and difference computation in <b>params.py</b>. We default to the ones used in our paper.

- <b>Input optimization</b>: The main input optimization code can be found in <b>main.py</b>. The training hyper-parameters and transform functions T in params.py can be defined as needed. The resultant transformed and projected input-output pairs will be dumped into the log directory. <b>Please see input_optimize() for details on the optimization process.</b>

- <b>Difference computation</b>: With the resultant components from the input optimization in main.py, the corresponding differences between them can be computed for model behavior analysis in <b>compute_differences.py</b>. The resultant difference vectors <b>d</b> per input will be dumped into the same log directory as before.

- <b>Dimensionality reduction</b>: A global snapshot of the model behavior per transform function or instance can be obtained via dimensionality reduction of the difference vectors per input in <b>dim_reduction_plot.py</b>.

## Citation
If you use this code for your research, please cite our IEEE TVCG paper. 
```
@article{prasad2023proactiv,
  title={ProactiV: Studying Deep Learning Model Behavior under Input Transformations},
  author={Prasad, Vidya and van Sloun, Ruud and Vilanova, Anna and Pezzotti, Nicola},
  year={2023},
  publisher={IEEE Transactions on Visualization and Computer Graphics},
  doi=10.1109/TVCG.2023.3301722
}
```

The paper is also publicly available on techrxiv: https://doi.org/10.36227/techrxiv.21946712

## Acknowledgments
SpinalNet model code: Original paper source code [SpinalNet](https://github.com/dipuk0506/SpinalNet/blob/master/MNIST_VGG/EMNIST_letters_VGG_and%20_SpinalVGG.py).

EMNIST dataset source: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373
