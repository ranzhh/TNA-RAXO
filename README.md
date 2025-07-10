<div align="center">

<h1> <a style="color:white; font-weight:bold;" href="https://pagf188.github.io/RAXO/">Superpowering Open-Vocabulary Object Detectors for X-ray Vision</a></h1>
<h2>ICCV 25</h2>

[Pablo Garcia-Fernandez](https://scholar.google.es/citations?user=xbtLSCcAAAAJ&hl=es),
[Lorenzo Vaquero](https://scholar.google.es/citations?user=G0ZcGDYAAAAJ&hl=es&oi=sra),
[Mingxuan Liu](https://scholar.google.com/citations?user=egL5-LsAAAAJ&hl=en),
[Feng Xue](https://scholar.google.com/citations?user=66SeiQsAAAAJ&hl=zh-CN),
[Daniel Cores](https://scholar.google.com/citations?user=pJqkUWgAAAAJ&hl=es)
[Nicu Sebe](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en)
[Manuel Mucientes](https://scholar.google.com/citations?user=raiz6p4AAAAJ)
[Elisa Ricci](https://scholar.google.com/citations?user=xf1T870AAAAJ&hl=en)


[![arXiv](https://img.shields.io/badge/cs.CV-2410.07752-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2503.17071)
[![GitHub](https://img.shields.io/badge/GitHub-RAXO-blue?logo=github)](https://github.com/PAGF188/RAXO)
[![Static Badge](https://img.shields.io/badge/website-RAXO-8A2BE2)](https://pagf188.github.io/RAXO/)
[![Static Badge](https://img.shields.io/badge/DET-COMPASS-8A2BE2)](https://huggingface.co/datasets/PAGF/DET-COMPASS)


</div>


## TO DO
- [x] Release dataset
- [ ] Release code
- [ ] Release precomputed visual descriptors

## Summary
This is the official repository of [**Superpowering Open-Vocabulary Object Detectors for X-ray Vision**](https://arxiv.org/abs/2503.17071).

The dataset is available at https://huggingface.co/datasets/PAGF/DET-COMPASS.

We propose **RAXO**, a training-free method that seamlessly adapts RGB OvOD models to X-ray

<div align="center">
  <img src="./assets/teaser.png" alt="Teaser">
</div>


## 📚 Abstract

Open-vocabulary object detection (OvOD) is set to revolutionize security screening by enabling systems to recognize any item in X-ray scans.
However, developing effective OvOD models for X-ray imaging presents unique challenges due to data scarcity and the modality gap that prevents direct adoption of RGB-based solutions.
To overcome these limitations, we propose **RAXO**, a training-free framework that repurposes off-the-shelf RGB OvOD detectors for robust X-ray detection.
RAXO builds high-quality X-ray class descriptors using a dual-source retrieval strategy.
It gathers relevant RGB images from the web and enriches them via a novel X-ray material transfer mechanism, eliminating the need for labeled databases.
These visual descriptors replace text-based classification in OvOD, leveraging intra-modal feature distances for robust detection.
Extensive experiments demonstrate that RAXO consistently improves OvOD performance, providing an average mAP increase of up to 17.0 points over base detectors.
To further support research in this emerging field, we also introduce DET-COMPASS, a new benchmark featuring bounding box annotations for over 300 object categories, enabling large-scale evaluation of OvOD in X-ray.
Code and dataset will be made available.
