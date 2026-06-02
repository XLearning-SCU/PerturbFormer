# PerturbFormer

This repository is the official implementation of the paper **"PerturbFormer establishes a virtual cell reference system for perturbation-attributable discovery"**.

PerturbFormer is a transformer-based foundation model for single-cell perturbation response prediction. It learns a **virtual cell reference system** that enables:

- **Perturbation response prediction** — given a control cell state and a perturbation (genetic or chemical), predict the resulting gene expression profile
- **Pathway-level interpretability** — the model's attention mechanism operates over biological pathways, providing insight into which pathways mediate the perturbation effect
- **Drug discovery simulation** — a *in silico-in vivo* closed-loop workflow that iteratively proposes and validates drug candidates using GSEA-based efficacy scoring

The model supports **both genetic perturbations** and **chemical perturbations** (small molecules).

## Repository Structure

```
├── config/                          # YAML configuration files
│   ├── pretrain_H800.yaml
│   ├── finetune.yaml
│   ├── lab_in_the_loop_simulation.yaml
│   └── split_config.toml            # Data split definitions
├── dataset_loader/                  # Data loading & preprocessing
│   ├── AnnCollectionDataLoader.py   # AnnData collection dataloader
│   ├── build.py                     # Dataloader builder
│   ├── data_utils.py                # Data structures & perturbation types
│   └── dataset_metadata.py          # Dataset registry & metadata
├── model/                           # Model implementation
│   ├── perturb_former.py            # Main PerturbFormer model
│   ├── modeling_transformer.py      # Transformer layer implementation
│   ├── modeling_utils.py            # CountMLP, CountEmbedding, LoRA, etc.
│   ├── PerturbFormerConfig.py       # Configuration dataclass & model gallery
│   └── PerturbPreprocessor.py       # Input preprocessing & embedding lookups
├── script/                          # Slurm launch scripts
│   ├── launch_pretrain.sh           # Multi-node pretraining launcher
│   ├── launch_finetune.sh           # fine-tuning launcher
│   ├── launch_lab_in_the_loop.sh    # Lab-in-the-loop simulation launcher
│   └── extract_smile_embedding.sh   # SMILES embedding precomputation
├── utils/                           # Utility modules
│   ├── args_parser.py
│   ├── iotools.py
│   ├── optim_utils.py
│   ├── eval_utils.py                # Evaluation metrics
│   ├── misc.py
│   ├── pdex/
│   └── cell_eval/                   # Optimized cell-eval toolkit
├── main_train.py
├── main_eval.py
├── engine_train.py                  # Training & evaluation engine
├── extract_smile_embedding.py       # Precompute UniMol SMILES embeddings
├── lab_in_the_loop.py               # Closed-loop drug discovery simulation
├── gsea_analysis.py
└── run_gsea.py
```

## Installation

```bash
# Create a Python virtual environment (Python >= 3.10)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The environment requires PyTorch 2.9+ (CUDA 12.6), HuggingFace Accelerate, and common scientific Python packages (scanpy, anndata, pandas, numpy, etc.). See [requirements.txt](requirements.txt) for the full list.

Additional dependencies for specific workflows:
- **SMILES embedding**: `unimol-tools` (for UniMol v2 molecular embeddings)
- **GSEA analysis**: `gseapy`, `blitzgsea`

## Data Preparation

Data is expected as **AnnData (`.h5ad`)** files organized by dataset. The project uses two data directories:

- **`data_dir`** — preprocessed single-cell perturbation data (count matrices, cell metadata)
- **`aux_data_dir`** — auxiliary data (drug metadata, precomputed gene/drug embeddings, gene metadata)

### Required AnnData Fields

Each `.h5ad` file in the training/test datasets must contain:

| Field | Description |
|-------|-------------|
| `.X` | Gene expression matrix (raw counts), cells × genes |
| `.var_names` | Gene Ensembl IDs (matching the gene vocabulary) |
| `.obs['perturb_name']` | Perturbation identifier (e.g., gene name for KD, `"[('drug_name', dose, unit)]"` for drugs) |
| `.obs['cell_type']` | Cell type / cell line name |
| `.obs['dataset_name']` | Dataset name (matching keys in `dataset_metadata.py`) |
| `.obs['batch']` | Batch identifier for control matching |
| `.obs['tscp_count']` | Total UMI count per cell |
| `.obs['median_umi']` | Median UMI count (used as target sum for normalization) |

### Auxiliary Data Files

| File | Location | Description |
|------|----------|-------------|
| `drug_metadata.csv` | `aux_data_dir` | Drug names, canonical SMILES, and metadata |
| `drug_embeddings.pt` | `aux_data_dir` | Precomputed SMILES → embedding dict (UniMol v2) |
| `target_gene_embeddings.pt` | `aux_data_dir` | Precomputed gene identifier → embedding dict |
| `gene_var.parquet` | `data_dir` | Gene metadata (Ensembl ID, symbol, etc.) |
| `HVG_pathway_graph_hvg_{N}.h5ad` | `data_dir` | Pathway-gene incidence matrix for highly variable genes |
| `{dataset_name}/perturb_mean.h5ad` | `data_dir` | Per-dataset precomputed perturbation mean expression |
| `{dataset_name}/padding_mask.h5ad` | `data_dir` | Per-dataset gene padding mask (for variable gene spaces) |

## Data Preprocessing

Before training, several preprocessing steps are required to build the necessary auxiliary data.

### 1. Gene Embedding Precomputation

Precompute gene perturbation embeddings from gene identifiers. Each gene (or gene combination) is mapped to a fixed-dimensional vector:
```bash
# Gene embeddings are stored in aux_data_dir/target_gene_embeddings.pt
# This is a dict {gene_name: tensor} loaded at runtime by GenePerturbEmbedding
```

### 2. Drug Embedding Extraction

Chemical perturbations (small molecules) are encoded via **UniMol v2** (570M parameter molecular foundation model). SMILES strings are precomputed into embeddings offline:

```bash
sbatch script/extract_smile_embedding.sh
# or run directly:
python extract_smile_embedding.py --config config/pretrain_H800.yaml --batch-size 32
```

The output is `drug_embeddings.pt` — a dict mapping canonical SMILES strings to embedding tensors, placed in `aux_data_dir`. At runtime, `DrugPerturbEmbedding` maps drug names → SMILES → precomputed embeddings.

### 3. Pathway Graph Construction

PerturbFormer uses a **pathway-gene incidence matrix** (`HVG_pathway_graph_hvg_{N}.h5ad`) that defines which genes belong to which biological pathways. This matrix is loaded at model initialization and used as an attention bias in the pathway aggregator. It should contain:

- `.X` — binary or weighted matrix of shape `[n_pathways, n_genes]`, where entry `[i, j]` indicates gene `j` belongs to pathway `i`
- `.var_names` — gene Ensembl IDs matching the HVG space
- `.obs_names` — pathway identifiers

### 4. Dataset-Specific Artifacts

For each dataset, precompute the following and place them under `data_dir/{dataset_name}/`:

| File | Description |
|------|-------------|
| `perturb_mean.h5ad` | Per-perturbation mean expression vectors (rows = perturbation keys, columns = genes) |
| `padding_mask.h5ad` | Boolean gene mask for datasets using a subset of the full gene space |

These are loaded on demand by `PerturbPreprocessor` at training time.

## Usage

### Pretraining

Pretrain the model from scratch on large-scale single-cell perturbation data:

```bash
# Single-node (4 GPUs)
accelerate launch --num_processes 4 --mixed_precision bf16 \
    main_train.py --config config/pretrain_H800.yaml

# Multi-node via Slurm
sbatch script/launch_pretrain.sh
```

### Fine-tuning

Fine-tune a pretrained model on downstream datasets:

```bash
# Single dataset
accelerate launch --num_processes 4 --mixed_precision bf16 \
    main_train.py --config config/finetune.yaml

# Batch fine-tuning across datasets
sbatch script/launch_finetune.sh
```

Fine-tuning supports several strategies:
- **`standard`（default）** — freeze the gene/drug perturbation encoders and expression encoder; fine-tune the pathway encoder, perturbation decoder, cell interactor, and prediction head
- **`full`** — fine-tune all model parameters
- **`head_only`** — only fine-tune the final prediction head (suitable for very small datasets)
- **`norm_tuning`** — freeze backbone, fine-tune all normalization layers and prediction head (best for domain adaptation)
- **`LoRA`** — parameter-efficient fine-tuning via low-rank adaptation (applied to attention projections and MLP gates)

### Evaluation

Run evaluation with a pretrained/fine-tuned model:

```bash
accelerate launch --num_processes 4 --mixed_precision bf16 \
    main_eval.py --config path/to/config.yaml
```

The evaluation outputs include:
- Predicted perturbation expression profiles (saved as `.h5ad`)
- Attention weights (gene↔pathway, pathway↔pathway)
- Evaluation metrics (Pearson correlation, Spearman correlation, R², etc.)

### Lab-in-the-Loop Drug Discovery Simulation

The closed-loop simulation iteratively proposes drug candidates, validates them via GSEA, and retrains:

```bash
sbatch script/launch_lab_in_the_loop.sh
```

Each round:
1. Selects top-K candidate perturbations based on predicted expression change
2. Runs GSEA to score pathway enrichment (e.g., apoptosis, proliferation)
3. Checks against ground-truth effective drugs (oracle stopping criterion)
4. Adds selected candidates to the training pool for the next round

Configuration in [lab_in_the_loop.py](lab_in_the_loop.py):
- `RANK_RULE` — gene ranking method: `signed_log10_fdr`
- `CONTINUAL_TRAINING` — whether each round continues from the previous round's last model
- `TOP_K` — number of candidates proposed per round

## Citation

Citation information will be added with the public release.

## License

[To be added with public release.]
