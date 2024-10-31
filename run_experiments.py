import numpy as np
import pandas as pd
import warnings
import os
import matplotlib.pyplot as plt
import sys
from datetime import datetime
from datasets import load_dataset
from lambeq import BobcatParser, Rewriter, RemoveCupsRewriter
from lambeq import AtomicType, IQPAnsatz, MPSAnsatz, Sim14Ansatz, Sim15Ansatz, SpiderAnsatz, StronglyEntanglingAnsatz, TensorAnsatz
from lambeq.backend.tensor import Dim
from pytket.extensions.qiskit import AerBackend
from lambeq import TketModel, QuantumTrainer, SPSAOptimizer, BinaryCrossEntropyLoss, Dataset

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Define logging function for model training information
def write_log(log, ansatz_name, rewriter_name):
    log_path = f"{log_dir}/{ansatz_name}_{rewriter_name}.log"
    with open(log_path, "a") as f:
        f.write(f"{str(log)}\n")
    print(log)

# Define data reading function
def read_data(filename):
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            t = int(line[0])
            labels.append([t, 1 - t])
            sentences.append(line[1:].strip())
    return labels, sentences

# Define rewriter functions
def re(diagrams):
    rewriter = Rewriter(['prepositional_phrase', 'determiner'])
    return [rewriter(diagram) for diagram in diagrams]

def re_norm(diagrams):
    rewriter = Rewriter(['prepositional_phrase', 'determiner'])
    rewritten_diagrams = [rewriter(diagram) for diagram in diagrams]
    return [diagram.normal_form() for diagram in rewritten_diagrams]

def re_norm_cur(diagrams):
    rewriter = Rewriter(['prepositional_phrase', 'determiner'])
    rewritten_diagrams = [rewriter(diagram) for diagram in diagrams]
    curry_functor = Rewriter(['curry'])
    return [curry_functor(diagram.normal_form()) for diagram in rewritten_diagrams]

def re_norm_cur_norm(diagrams):
    rewriter = Rewriter(['prepositional_phrase', 'determiner'])
    rewritten_diagrams = [rewriter(diagram) for diagram in diagrams]
    curry_functor = Rewriter(['curry'])
    return [curry_functor(diagram.normal_form()).normal_form() for diagram in rewritten_diagrams]

# Define training function
def train_model(all_circuits, train_circuits, dev_circuits, rewriter, ansatz):
    model = TketModel.from_diagrams(all_circuits, backend_config=backend_config)
    loss = BinaryCrossEntropyLoss()
    acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2

    EPOCHS = 120
    BATCH_SIZE = 32

    trainer = QuantumTrainer(
        model,
        loss_function=loss,
        epochs=EPOCHS,
        optimizer=SPSAOptimizer,
        optim_hyperparams={'a': 0.05, 'c': 0.06, 'A': 0.01 * EPOCHS},
        evaluate_functions={'acc': acc},
        evaluate_on_train=True,
        verbose='text',
        seed=0
    )

    train_dataset = Dataset(train_circuits, train_labels, batch_size=BATCH_SIZE)
    val_dataset = Dataset(dev_circuits, dev_labels, shuffle=False)

    trainer.fit(train_dataset, val_dataset, log_interval=1)
    return trainer.train_epoch_costs, trainer.val_costs, trainer.train_eval_results['acc'], trainer.val_eval_results['acc']

# Run a single experiment
def run_single_experiment(ansatz_index, rewriter_index):
    ansatz = ansatzes[ansatz_index]
    rewriter = rewriters[rewriter_index]
    ansatz_name = type(ansatz).__name__
    rewriter_name = rewriter.__name__

    # Prepare directory structure
    plot_dir = os.path.join(log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Log the start
    write_log(f"Starting training for {ansatz_name} with {rewriter_name}", ansatz_name, rewriter_name)

    # Preprocess data
    train_diagrams = rewriter(raw_train_diagrams)
    dev_diagrams = rewriter(raw_dev_diagrams)
    test_diagrams = rewriter(raw_test_diagrams)

    train_circuits = [ansatz(diagram) for diagram in train_diagrams]
    dev_circuits = [ansatz(diagram) for diagram in dev_diagrams]
    test_circuits = [ansatz(diagram) for diagram in test_diagrams]
    all_circuits = train_circuits + dev_circuits + test_circuits

    # Train model and retrieve losses and accuracies
    train_loss, val_loss, train_acc, val_acc = train_model(all_circuits, train_circuits, dev_circuits, rewriter, ansatz)

    # Log results
    write_log("Train Loss:", ansatz_name, rewriter_name)
    write_log(train_loss, ansatz_name, rewriter_name)
    write_log("Validation Loss:", ansatz_name, rewriter_name)
    write_log(val_loss, ansatz_name, rewriter_name)
    write_log("Train Accuracy:", ansatz_name, rewriter_name)
    write_log(train_acc, ansatz_name, rewriter_name)
    write_log("Validation Accuracy:", ansatz_name, rewriter_name)
    write_log(val_acc, ansatz_name, rewriter_name)

    # Plot results
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].plot(train_loss, label="Training Loss")
    axs[0].plot(val_loss, label="Validation Loss")
    axs[0].set_title(f"Loss for {ansatz_name} with {rewriter_name}")
    axs[0].legend()

    axs[1].plot(train_acc, label="Training Accuracy")
    axs[1].plot(val_acc, label="Validation Accuracy")
    axs[1].set_title(f"Accuracy for {ansatz_name} with {rewriter_name}")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{ansatz_name}_{rewriter_name}_loss_accuracy.png")
    plt.close(fig)

if __name__ == '__main__':
    current_time = sys.argv[3]
    log_dir = os.path.join('/scratch/jd5018/qlang/logs', current_time)
    os.makedirs(log_dir, exist_ok=True)
    
    # Load data and initialize parser
    train_labels, train_data = read_data('datasets/mc_train_data.txt')
    dev_labels, dev_data = read_data('datasets/mc_dev_data.txt')
    test_labels, test_data = read_data('datasets/mc_test_data.txt')

    parser = BobcatParser(verbose='text')
    raw_train_diagrams = parser.sentences2diagrams(train_data)
    raw_dev_diagrams = parser.sentences2diagrams(dev_data)
    raw_test_diagrams = parser.sentences2diagrams(test_data)

    rewriters = [re, re_norm, re_norm_cur, re_norm_cur_norm]
    ansatzes = [
        IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=2, n_single_qubit_params=3),
        StronglyEntanglingAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=2, n_single_qubit_params=3),
        Sim14Ansatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=2, n_single_qubit_params=3),
        Sim15Ansatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=2, n_single_qubit_params=3)
    ]

    shots = 8192
    backend = AerBackend()
    backend_config = {
        'backend': backend,
        'compilation': backend.default_compilation_pass(2),
        'shots': shots
    }

    # Parse indices for Ansatz and Rewriter from command line arguments
    ansatz_index = int(sys.argv[1])
    rewriter_index = int(sys.argv[2])

    # Run the single experiment
    run_single_experiment(ansatz_index, rewriter_index)
