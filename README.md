# MFL
A PyTorch implementation of MFL: A Multi-modal Fully Hyperbolic Learning Framework for Fake and AI-generated News Detection

![](https://github.com/GreatL/MFL/raw/main/MFL.png)

## Abstract
With the rapid growth of mobile network and Artificially Intelligence Generated Content (AIGC), the spread of fake news has intensified, making the identification of credible information increasingly challenging. Most existing detection methods focus on binary classification (e.g., real vs. fake or
human vs. AI-generated) and rely on Euclidean representations, which fail to capture the complex and hierarchical relationships among news articles. To overcome these limitations, a novel Multi-modal Fully Hyperbolic Learning (MFL) framework is introduced for three-class fake news detection: real news, fake news, and AI-generated news. The framework models news as a graph to capture interrelations among posts and leverages pre-trained models to extract text and image features. A crossattention mechanism fuses textual and visual features, capturing semantic associations between modalities. These features are further integrated using a Lorentz-based multi-modal fusion module to enhance representation quality. A fully hyperbolic graph neural network learns hierarchical and relational structures in hyperbolic space, supported by a hyperbolic information embedding algorithm. Experiments on four real-world datasets demonstrate that MFL outperforms state-of-the-art methods, effectively tackling the complexities of multi-class fake news detection.

## Datasets
The datasets used in this paper are available for download on the public web. In the `data` folder of this repository, you can find the download links for the preprocessed data, which can be used by researchers to reproduce the code.

## Usage
Before training, run

```source set_env.sh```

This will create environment variables that are used in the code.

To run the experiments, simply download the datasets and put them in the `data` directory. Then run the corresponding training script, e.g.,

```bash run.sh```


