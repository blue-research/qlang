import numpy as np
import pandas as pd
import warnings
import os
import matplotlib.pyplot as plt
from datasets import load_dataset
from lambeq import BobcatParser, Rewriter, RemoveCupsRewriter
from lambeq import AtomicType, IQPAnsatz, MPSAnsatz, Sim14Ansatz, Sim15Ansatz, SpiderAnsatz, StronglyEntanglingAnsatz, TensorAnsatz
from lambeq.backend.tensor import Dim
from pytket.extensions.qiskit import AerBackend
from lambeq import TketModel, QuantumTrainer, SPSAOptimizer, BinaryCrossEntropyLoss, Dataset

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def write_log(log, ansatz_name, rewriter_name):
    print(str(log))
    with open(f"./logs/{ansatz_name}_{rewriter_name}.log", "w") as f:
        f.write(str(log))
        
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
    rewritten_diagrams = [rewriter(diagram) for diagram in diagrams]
    return rewritten_diagrams

def re_norm(diagrams):
    rewriter = Rewriter(['prepositional_phrase', 'determiner'])
    rewritten_diagrams = [rewriter(diagram) for diagram in diagrams]
    normalised_diagrams = [diagram.normal_form() for diagram in rewritten_diagrams]
    return normalised_diagrams

def re_norm_cur(diagrams):
    rewriter = Rewriter(['prepositional_phrase', 'determiner'])
    rewritten_diagrams = [rewriter(diagram) for diagram in diagrams]
    normalised_diagrams = [diagram.normal_form() for diagram in rewritten_diagrams]
    curry_functor = Rewriter(['curry'])
    curried_diagrams = [curry_functor(diagram) for diagram in normalised_diagrams]
    return curried_diagrams

def re_norm_cur_norm(diagrams):
    rewriter = Rewriter(['prepositional_phrase', 'determiner'])
    rewritten_diagrams = [rewriter(diagram) for diagram in diagrams]
    normalised_diagrams = [diagram.normal_form() for diagram in rewritten_diagrams]
    curry_functor = Rewriter(['curry'])
    curried_diagrams = [curry_functor(diagram) for diagram in normalised_diagrams]
    normalised_diagrams2 = [diagram.normal_form() for diagram in curried_diagrams]
    return normalised_diagrams2

# Define training function
def train_model(all_circuits, train_circuits, dev_circuits, rewriter, ansatz):
    model = TketModel.from_diagrams(all_circuits, backend_config=backend_config)
    loss = BinaryCrossEntropyLoss()
    acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2  # half due to double-counting

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
    return trainer.train_epoch_costs, trainer.val_costs

# Experiment and plot saving function
def run_experiment_with_saving_plots():
    for i, ansatz in enumerate(ansatzes):
        ansatz_name = type(ansatz).__name__
        fig, axs = plt.subplots(1, len(rewriters), figsize=(20, 5), sharex=True, sharey=True)

        for j, rewriter in enumerate(rewriters):
            rewriter_name = rewriter.__name__
            train_diagrams = rewriter(raw_train_diagrams)
            dev_diagrams = rewriter(raw_dev_diagrams)
            test_diagrams = rewriter(raw_test_diagrams)

            train_circuits = [ansatz(diagram) for diagram in train_diagrams]
            dev_circuits = [ansatz(diagram) for diagram in dev_diagrams]
            test_circuits = [ansatz(diagram) for diagram in test_diagrams]
            all_circuits = train_circuits + dev_circuits + test_circuits

            # Train model and retrieve losses
            write_log(f"Training with {ansatz_name} and {rewriter_name}", ansatz_name, rewriter_name)
            train_loss, val_loss = train_model(all_circuits, train_circuits, dev_circuits, rewriter, ansatz)
            write_log("Train loss:", ansatz_name, rewriter_name)
            write_log(train_loss, ansatz_name, rewriter_name)
            write_log("Validation loss:", ansatz_name, rewriter_name)
            write_log(val_loss, ansatz_name, rewriter_name)
            
            # Plotting losses on the designated subplot
            axs[j].plot(train_loss, label="Training Loss")
            axs[j].plot(val_loss, label="Validation Loss")
            axs[j].set_title(f"{ansatz_name} with {rewriter_name}")
            axs[j].legend()

        # Save the plot for the current Ansatz with all rewriters as a separate file
        plt.suptitle(f"Loss Curves for {ansatz_name}")
        filename = f"plots/{ansatz_name}_loss_plots.png"
        plt.savefig(filename)
        plt.close(fig)

# Main program
if __name__ == '__main__':
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

    run_experiment_with_saving_plots()
