<br />
<div align="center">
  
  <h1 align="center">Just Noticeable Robust Attack (JNRA)</h1>
  <img src="https://github.com/simone-dotolo/jnra/blob/main/media/demo.png" width=800>

</div>

<!-- ABOUT THE PROJECT -->
## About The Project

The Just Noticeable Robust Attack is a novel approach designed to craft adversarial perturbations that are both imperceptible to human observers and robust against image purification defenses. This approach strategically inserts strong perturbations into regions of an image where they are less likely to be perceived by a human observer, using a simple Just Noticeable Difference (JND) model.
<br />
<div align="center">
  <img src="https://github.com/simone-dotolo/jnra/blob/main/media/jndscaling.png" width=800>
  <img src="https://github.com/simone-dotolo/jnra/blob/main/media/jnra.png" width=800>
  <img src="https://github.com/simone-dotolo/jnra/blob/main/media/jndthresholding.png" width=800>
</div>

## Installation

1. Clone the repo:
   ```sh
   git clone https://github.com/simone-dotolo/jnra.git
   ```
2. Create the virtual environment:
   ```sh
   conda env create -n jnra -f environment.yml
   ```

## Usage

1. To protect the artworks with JNRA:
   ```sh
   python3 jnra_protect.py
   ```
2. To purify the protected artworks:
   ```sh
   python3 apply_purification.py --data_path data/wikiart_zdzislaw_beksinki/protected/jnra --purification jpeg --device cuda
   ```
3. To finetune the model:
   ```sh
   python3 finetune.py --in_dir data/wikiart_zdzislaw_beksinki/protected_purified/jnra/jpeg/train --out_dir models/wikiart_zdzislaw-beksinki/protected_purified/jnra/jpeg
   ```
4. To generate artworks:
   ```sh
   python3 generate.py --in_dir models/wikiart_zdzislaw-beksinki/protected_purified/jnra/jpeg --out_dir generated_images/wikiart_rene-magritte/protected_purified/jnra/jpeg --prompts prompts/wikiart_zdzislaw-beksinki.txt
   ```
5. To evaluate the protection:
   ```sh
   python3 evaluate.py --original_path generated_images/wikiart_rene-magritte/original --generated_path generated_images/wikiart_rene-magritte/protected_purified/jnra/jpeg --device cuda:0
   ```
   
<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
- Finetuning and generation scripts: https://github.com/ethz-spylab/robust-style-mimicry/
- DiffJPEG: https://github.com/mlomnitz/DiffJPEG
- CMMD: https://github.com/sayakpaul/cmmd-pytorch

<!-- CONTACT -->
## Contact

Simone Dotolo - sim.dotolo@gmail.com

LinkedIn: [https://www.linkedin.com/in/simone-dotolo/](https://www.linkedin.com/in/simone-dotolo/)
