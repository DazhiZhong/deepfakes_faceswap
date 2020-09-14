# Faceswap Adversarial Attack
⚠️ This project is a forked version of [this repo](https://github.com/joshua-wu/deepfakes_faceswap)! Please go there for the original code.

![examples faceswap singular small](README/examples%20faceswap%20singular%20small.png)

In our paper, we propose AI-FGSM, a method for transferable adversarial attack for a wide range of models. We experiment of three deep neural networks consisting of image translation and image classification networks. This is our third experiment, which is done on face swapping networks. See our previous work at  [this repo](https://github.com/DazhiZhong/disrupting-deepfakes) (StarGAN) and [this repo](https://github.com/jasonliuuu/SI-AI-FGSM) (Inception models).

![examples truepcas](README/examples%20truepcas.png)

Fig 2: randomly chosen attacked images and their reference images.



Google Drive shared file for the entire repo, training data, and trained models: https://drive.google.com/file/d/1-2bNmny7Xlo818mqPCysw54C09FEyp4f/view?usp=sharing



**Requirements:**

    Python 3
    Opencv 3
    Tensorflow 1.3+(?)
    Keras 2

**How to run:**

    python attack.py

Alternatively, use the jupyter notebook in the files, or colab notebook [here](https://colab.research.google.com/drive/1Ru4lY-WuH5V0LdxBh4YwvJe8qeclSKko?usp=sharing)