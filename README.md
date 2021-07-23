# FG2021-BoLD

Code for reproducing our proposed state-of-the-art model, for categorical and continuous emotion recognition on the basis of the newly assembled and challenging Body Language Dataset (BoLD), as submitted to the 16th IEEE International Conference on Automatic Face and Gesture Recognition (FG2021)

### Preparation

* Download the [BoLD dataset](https://cydar.ist.psu.edu/emotionchallenge/index.php).
* Use [https://github.com/yjxiong/temporal-segment-networks](https://github.com/yjxiong/temporal-segment-networks) in order to extract RGB and Optical Flow for the dataset.
* Change the directories in "dataset.py", "skeleton_dataset.py" and paths for pre-trained weights in "models.py" files.

### Training

Train an Temporal Segment Network using the RGB modality (TSN-RGB) and only the body stream, on the BoLD dataset:

> python train_tsn.py --config main_config.json --exp_name TSN_RGB_b --modality RGB --device 0 --rgb_body  

Add context branch:

> python train_tsn.py --config main_config.json --exp_name TSN_RGB_bc --modality RGB --device 0 --context --rgb_body --rgb_context

Add visual embedding loss:

> python train_tsn.py --config main_config.json --exp_name TSN_RGB_bc_embed --modality RGB --device 0 --context --rgb_body --rgb_context --embed

Initialize body stream with ImageNet pre-trained weights:

> python train_tsn.py --config main_config.json --exp_name TSN_RGB_bc_embed --modality RGB --device 0 --context --rgb_body --rgb_context --embed --pretrained_imagenet

Change modality to Optical Flow (all streams initialized with ImageNet pre-trained weights):

> python train_tsn.py --config main_config.json --exp_name TSN_Flow_bc_embed --modality Flow --device 0 --context --flow_body --flow_context --embed --pretrained_imagenet

### Pretrained Models

We also offer weights of an TSN-RGB with body, context, face, scenes and attributes streams, embedding loss and partial batch normalization (0.2590 validation ERS), a TSN-Flow model with body, context and face streams, embedding loss and partial batch normalization (0.2408 validation ERS) and a fine-tuned ST-GCN model with spatial labeling strategy and pre-training on Kinetics (0.2237 validation ERS). Their fusion achieves an ERS of 0.2613 on the validation set. You can download the pretrained models [here](https://ntuagr-my.sharepoint.com/:f:/g/personal/filby_ntua_gr/EkFAi_QSn9NDsFTylvoAJrQBuvh6eQWkbgTuZcyMWWPR2w?e=xxw6h9).

Moreover, all Places365 and SUN pre-trained models that were utilized in our experiments can be found [here](https://github.com/CSAILVision/places365). ImageNet pre-trained weights can be found [here]. Lastly, we do not intend to release our AffectNet pre-trained ResNet-18 and ResNet-50 variants. 

### Inference

Run inference on the BoLD test set (using the provided pre-trained models): 

> python infer_tsn.py --modality RGB --device 0 --context --rgb_body --rgb_context --rgb_face --scenes --attributes --context --embed --partial_bn --checkpoint {path to .pth checkpoint file} --save_outputs --output_dir {directory to save outputs to} --exp_name some_name --mode test

### Citation

If you use our code for your research, consider citing the following papers:

```
@misc{pikoulis2021leveraging,
      title={Leveraging Semantic Scene Characteristics and Multi-Stream Convolutional Architectures in a Contextual Approach for Video-Based Visual Emotion Recognition in the Wild}, 
      author={Ioannis Pikoulis and Panagiotis P. Filntisis and Petros Maragos},
      year={2021},
      eprint={2105.07484},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{NTUA_BEEU,
  title={Emotion Understanding in Videos Through Body, Context, and Visual-Semantic Embedding Loss},
  author={Filntisis, Panagiotis Paraskevas and Efthymiou, Niki and Potamianos, Gerasimos and Maragos, Petros},
  booktitle={ECCV Workshop on Bodily Expressed Emotion Understanding},
  year={2020}
}
```

### Acknowlegements

* [https://github.com/filby89/NTUA-BEEU-eccv2020](https://github.com/filby89/NTUA-BEEU-eccv2020)
* [https://github.com/victoresque/pytorch-template](https://github.com/victoresque/pytorch-template)
* [https://github.com/yjxiong/tsn-pytorch](https://github.com/yjxiong/tsn-pytorch)
* [https://github.com/yysijie/st-gcn](https://github.com/yysijie/st-gcn)
* [https://github.com/CSAILVision/places365](https://github.com/CSAILVision/places365)

### Contact

For questions feel free to open an issue.