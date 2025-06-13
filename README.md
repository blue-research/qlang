# Comparative Study of the Ansätze in Quantum Language Models

This repository contains the implementation and experimental framework used in the study _"Comparative study of the ansätze in quantum language models"_. The research explores how different ansätze and hyperparameters influence Quantum Natural Language Processing (QNLP) models for text classification tasks. It evaluates both circuit-based and tensor-based approaches using the Lambeq library and quantum simulation backends.

## 🔬 Features

- Full QNLP pipeline with:
  - Sentence-to-diagram conversion via pregroup grammar
  - Optional rewriting of diagrams (`re`, `re_norm`, `re_norm_cur`, `re_norm_cur_norm`)
  - Conversion into quantum circuits or tensor networks
- Multiple ansätze supported:
  - **Circuit-based**:
    - IQPAnsatz
    - StronglyEntanglingAnsatz
    - Sim14Ansatz
    - Sim15Ansatz
  - **Tensor-based**:
    - MPSAnsatz
    - SpiderAnsatz
    - TensorAnsatz
- Hyperparameter exploration:
  - Number of layers
  - Single-qubit rotations
- Results include:
  - Training/validation loss and accuracy
  - Overfitting and convergence trends
  - Test performance comparison
 
## 🧪 Running the Experiments

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Training the models and hyperparameter tuning & evaluation

```
python run_experiments_all.py
```

## 📊 Results Summary

Best circuit-based performance: Sim14Ansatz + re_norm_cur_norm (100% validation accuracy)

Best tensor-based performance: SpiderAnsatz + re (100% validation accuracy, lowest loss)

Diagram simplification via rewriting significantly improves both convergence and generalization for grammatically simple sentences.

Careful hyperparameter tuning is key to optimal model design for the average sentence complexity of the dataset.

## 📚 Citation
If you use this repository or the results in your research, please cite:

```
@article{DelCastillo:2025edw,
    author = "Del Castillo, Jordi and Zhao, Dan and Pei, Zongrui",
    title = {{Comparative study of the ans\"atze in quantum language models}},
    eprint = "2502.20744",
    archivePrefix = "arXiv",
    primaryClass = "quant-ph",
    month = "2",
    year = "2025"
}
```

📬 Contact
For questions or collaborations, contact:

Jordi Del Castillo

Email: jordi.d@nyu.edu | jordi.delcastillo.1@gmail.com

Dan Zhao

Email: dz1158@nyu.edu

Zongrui Pei

Email: zp2137@nyu.edu | peizongrui@gmail.com
