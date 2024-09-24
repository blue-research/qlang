# qnlg - Quantum Natural Language Generation

## Tasks
1. Improve classification model's accuracy
    - Change dataset from mc_train -> stanfordnlp/sst2 (larger dataset) <-! currently on hold due to parser not being able to parse complex sentences
    - Test different ansatz and hyperparameters and rank them
    - Test different classification model and hyperparameters and rank them
2. Improve qnlg algorithm's accuracy (draft)
    - Superposition for Sentence Generation: Create a superposition of multiple sentence states to evaluate them simultaneously, leveraging quantum parallelism. This requires a quantum circuit that can represent and manipulate sentences in superposition.
    - Entanglement for Contextual Understanding: Design parts of the quantum circuit to entangle words or phrases, potentially capturing more nuanced, contextual relationships between them. This could lead to more accurate scoring and measure similarity of sentences.

## References
- https://github.com/gamatos/qnlp-binary-classification
- https://github.com/NEASQC/WP6_QNLP?tab=readme-ov-file

## Notes
1. Change mc_train to rc_train dataset
2. Run experiments with different ansatz and organize results
3. Current goal is to establish a relationship between different ansatz methods and sentence representation.
