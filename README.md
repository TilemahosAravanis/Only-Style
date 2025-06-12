<div align="center">

# Only-Style: Stylistic Cinsistency in Image Generation without Content Leakage 

Official implementation of "Only-Style: Stylistic Consistency in Image Generation without Content Leakage". 

</div>

## Installation

---

This repository provides instructions to replicate the `Only-Style` Conda environment. Follow the steps below to set up the environment on your local machine.

### Prerequisites

Ensure the following software is installed on your system:

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

---

### Steps to Install the Environment

1. **Clone or Download the Repository**

   Download or clone the repository that contains the `environment.yml` file.

2. **Create the Conda Environment**

   Run the following command to create the environment:
   ```bash
   conda env create -f environment.yml

## Inference

---

Use the following command to generate a stylistically aligned image pair without content leakage:

```bash
python Only_Style_CLI.py --seed <seed_value> --S_ref "<reference_sentence>" --ref_token "<reference_token>" --S_tgt "<target_sentence>" --tgt_token "<target_token>" --style "<style_description>" --precision <precision_value> --output_dir "<output_directory>"
```

---

Below is a breakdown of the arguments:

| Argument         | Description                                                                                           | Example Value              |
|-------------------|------------------------------------------------------------------------------------------------------|----------------------------|
| `--seed`         | Random seed for reproducibility.                                                                      | `42`                       |
| `--S_ref`        | Description of the stylistic reference subject.                                                       | `"A cat"`                  |
| `--ref_token`    | Token identifying the reference subject.                                                              | `"cat"`                    |
| `--S_tgt`        | Description of the target subject.                                                                    | `"A train"`                |
| `--tgt_token`    | Token identifying the target subject.                                                                 | `"train"`                  |
| `--style`        | Description of the desired style for the image.                                                       | `"in 3D rendering style"`  |
| `--precision`    | Precision of the binary search algorithm used to determine the optimal alpha (see manuscript).        | `0.03`                     |
| `--output_dir`   | Directory where the generated images and logs will be saved.                                          | `"./output"`               |

---

### Example

```bash
python Only_Style_CLI.py --seed 42 --S_ref "A cat" --ref_token "cat" --S_tgt "A train" --tgt_token "train" --style "in 3D rendering style" --precision 0.03 --output_dir "./output"
```

---

### Outputs

The inference script produces the following outputs in the directory specified by `--output_dir`:

1. **Reference Image (`Reference.png`)**:
   - Image used as a stylistic reference.

2. **Style-Aligned Target Image (`StyleAligned_Target.png`)**:
   - Target image aligned stylistically via [StyleAligned](https://style-aligned-gen.github.io/).

3. **Content Leakage Heatmap (`StyleAligned_Leakage.png`)**:
   - Visualization of content leakage, overlaid as a heatmap on the target image mentioned above.

4. **Optimized Target Image (`Only_Style_Target.png`)**:
   - Final image generated using controlled style alignment, avoiding content leakage.

5. **Log File (`log.txt`)**:
   - Contains details such as the number of leaky patches detected in every step of the binary search process for the optimal alpha.

---

## Code Development Status

- [x] Inference code for Only-Style
- [ ] Support for multi-image and multi-subject cases
- [ ] Support for real stylistic references  
- [ ] Support for Generalized Leakage Localization (input reference-target pair)  
- [ ] Evaluation code

---

## Acknowledgements

This work builds upon the [Style-Aligned](https://github.com/google/style-aligned) repository by Google Research. We thank the authors for making their code publicly available.
