# RAXO - T&A of CV Exploration
This project is an exploratory work on how to improve the RAXO pipeline by Fernandez et al. building on their original work which can be found at [this GitHub link](https://pagf188.github.io/RAXO/). The first commits will be mostly about fixing paths and improving code structure, in order to make it possible for all of us to work on this. Then we'll experiment with new proposal mechanisms for the MaterialTransfer part.

The project is developed by Riccardo Benevelli, Leonardo Bottona and Matteo Ranzetti as a part of ther Trends and Applications of Computer Vision course. The course itself is held by Prof. Giulia Boato and Prof. Massimiliano Mancini at the University of Trento.

## How can I get this to work?
Refer to the [original README](README_original.md) in order to know what assets you'll have to download, such as models and datasets. Then, take a look at the configuration variables in .env.example and make sure to set them in a .env file. If you're installing stuff locally (e.g. `sam2`) make sure to install it by hand in the project (e.g. `u.v. add /path/to/sam2`).

# SAM 2
SAM is needed for this project to work. In order to download it, clone the excellent repo over at `https://github.com/facebookresearch/sam2`, then install sam2 via uv. Make sure to download the checkpoints (check out `checkpoints/` in the sam2 repo, there's a script.) Point the SAM_PATH variable to the path where you cloned the sam2 repo.