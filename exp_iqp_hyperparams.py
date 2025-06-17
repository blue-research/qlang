import numpy as np
import pandas as pd
import warnings
import os
import sys
import matplotlib.pyplot as plt
from lambeq import BobcatParser, Rewriter, RemoveCupsRewriter
from lambeq import AtomicType, IQPAnsatz, MPSAnsatz, Sim14Ansatz, Sim15Ansatz, SpiderAnsatz, StronglyEntanglingAnsatz, TensorAnsatz
from lambeq.backend.tensor import Dim
from pytket.extensions.qiskit import AerBackend
from lambeq import TketModel, QuantumTrainer, SPSAOptimizer, BinaryCrossEntropyLoss, Dataset
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Define logging function for model training information
def write_log(log, n_layers, n_single_qubit_params):
    print(str(log)) #stdout
    log_path=f"{log_dir}/{n_layers}_{n_single_qubit_params}.log"
    with open(log_path, "a") as f:
        f.write(f"{str(log)}\n")
        
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
def train_model(all_circuits, train_circuits, dev_circuits):
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
    return trainer.train_epoch_costs, trainer.val_costs, trainer.train_eval_results['acc'], trainer.val_eval_results['acc']

def plot_experiment_results_separately(n_single_qubit_results, n_layers_results):
    # Plot for Experiment 1: Fixed n_layers=2, varying n_single_qubit_params
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    for n_single_qubit_params, train_loss, val_loss, train_acc, val_acc in n_single_qubit_results:
        axs[0].plot(train_loss, label=f"Train Loss (n_single={n_single_qubit_params})", linewidth=2)
        axs[0].plot(val_loss, linestyle='--', label=f"Val Loss (n_single={n_single_qubit_params})", linewidth=2)
        axs[1].plot(train_acc, label=f"Train Acc (n_single={n_single_qubit_params})", linewidth=2)
        axs[1].plot(val_acc, linestyle='--', label=f"Val Acc (n_single={n_single_qubit_params})", linewidth=2)

    axs[0].set_title("Loss (n_layers=2)", fontsize=16)
    axs[0].set_xlabel("Epochs", fontsize=14)
    axs[0].set_ylabel("Loss", fontsize=14)
    axs[0].legend(fontsize=10)
    axs[0].tick_params(axis='both', which='major', labelsize=12)

    axs[1].set_title("Accuracy (n_layers=2)", fontsize=16)
    axs[1].set_xlabel("Epochs", fontsize=14)
    axs[1].set_ylabel("Accuracy", fontsize=14)
    axs[1].legend(fontsize=10)
    axs[1].tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/loss_accuracy_n_single_qubit_params.png")
    plt.close(fig)

    # Plot for Experiment 2: Fixed n_single_qubit_params=3, varying n_layers
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    for n_layers, train_loss, val_loss, train_acc, val_acc in n_layers_results:
        axs[0].plot(train_loss, label=f"Train Loss (n_layers={n_layers})", linewidth=2)
        axs[0].plot(val_loss, linestyle='--', label=f"Val Loss (n_layers={n_layers})", linewidth=2)
        axs[1].plot(train_acc, label=f"Train Acc (n_layers={n_layers})", linewidth=2)
        axs[1].plot(val_acc, linestyle='--', label=f"Val Acc (n_layers={n_layers})", linewidth=2)

    axs[0].set_title("Loss (n_single_qubit_params=3)", fontsize=16)
    axs[0].set_xlabel("Epochs", fontsize=14)
    axs[0].set_ylabel("Loss", fontsize=14)
    axs[0].legend(fontsize=10)
    axs[0].tick_params(axis='both', which='major', labelsize=12)

    axs[1].set_title("Accuracy (n_single_qubit_params=3)", fontsize=16)
    axs[1].set_xlabel("Epochs", fontsize=14)
    axs[1].set_ylabel("Accuracy", fontsize=14)
    axs[1].legend(fontsize=10)
    axs[1].tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/loss_accuracy_n_layers.png")
    plt.close(fig)


def run_specific_experiments():
    # Experiment 1: Fixed n_layers=2, iterate over n_single_qubit_params
    fixed_n_layers = 2
    n_single_qubit_results = []
    for n_single_qubit_params in range(0, 5):
        ansatz = IQPAnsatz(
            {AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
            n_layers=fixed_n_layers,
            n_single_qubit_params=n_single_qubit_params,
        )
        train_circuits = [ansatz(diagram) for diagram in train_diagrams]
        dev_circuits = [ansatz(diagram) for diagram in dev_diagrams]
        test_circuits = [ansatz(diagram) for diagram in test_diagrams]
        all_circuits = train_circuits + dev_circuits + test_circuits

        train_loss, val_loss, train_acc, val_acc = train_model(all_circuits, train_circuits, dev_circuits)
        n_single_qubit_results.append((n_single_qubit_params, train_loss, val_loss, train_acc, val_acc))
        write_log("Train Loss:", fixed_n_layers, n_single_qubit_params)
        write_log(train_loss, fixed_n_layers, n_single_qubit_params)
        write_log("Validation Loss:", fixed_n_layers, n_single_qubit_params)
        write_log(val_loss, fixed_n_layers, n_single_qubit_params)
        write_log("Train Accuracy:", fixed_n_layers, n_single_qubit_params)
        write_log(train_acc, fixed_n_layers, n_single_qubit_params)
        write_log("Validation Accuracy:", fixed_n_layers, n_single_qubit_params)
        write_log(val_acc, fixed_n_layers, n_single_qubit_params)

    # Experiment 2: Fixed n_single_qubit_params=3, iterate over n_layers
    fixed_n_single_qubit_params = 3
    n_layers_results = []
    for n_layers in range(0, 5):
        ansatz = IQPAnsatz(
            {AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
            n_layers=n_layers,
            n_single_qubit_params=fixed_n_single_qubit_params,
        )
        train_circuits = [ansatz(diagram) for diagram in train_diagrams]
        dev_circuits = [ansatz(diagram) for diagram in dev_diagrams]
        test_circuits = [ansatz(diagram) for diagram in test_diagrams]
        all_circuits = train_circuits + dev_circuits + test_circuits

        train_loss, val_loss, train_acc, val_acc = train_model(all_circuits, train_circuits, dev_circuits)
        n_layers_results.append((n_layers, train_loss, val_loss, train_acc, val_acc))
        write_log("Train Loss:", n_layers, fixed_n_single_qubit_params)
        write_log(train_loss, n_layers, fixed_n_single_qubit_params)
        write_log("Validation Loss:", n_layers, fixed_n_single_qubit_params)
        write_log(val_loss, n_layers, fixed_n_single_qubit_params)
        write_log("Train Accuracy:", n_layers, fixed_n_single_qubit_params)
        write_log(train_acc, n_layers, fixed_n_single_qubit_params)
        write_log("Validation Accuracy:", n_layers, fixed_n_single_qubit_params)
        write_log(val_acc, n_layers, fixed_n_single_qubit_params)

    plot_experiment_results_separately(n_single_qubit_results, n_layers_results)
    return n_single_qubit_results, n_layers_results

def run_experiment():
    ansatz=IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=n_layers, n_single_qubit_params=n_single_qubit_params)
    try:
        train_circuits = [ansatz(diagram) for diagram in train_diagrams]
        dev_circuits = [ansatz(diagram) for diagram in dev_diagrams]
        test_circuits = [ansatz(diagram) for diagram in test_diagrams]
        all_circuits = train_circuits + dev_circuits + test_circuits
    except:
        print("Error with n_layers: ", n_layers, " and n_single_qubit_params: ", n_single_qubit_params)
        exit(1)

    write_log(f"Training with n_layers: {n_layers} and n_single_qubit_params: {n_single_qubit_params}", n_layers, n_single_qubit_params)
    
    train_loss, val_loss, train_acc, val_acc = train_model(all_circuits, train_circuits, dev_circuits, rewriter, ansatz)
    write_log("Train Loss:", n_layers, n_single_qubit_params)
    write_log(train_loss, n_layers, n_single_qubit_params)
    write_log("Validation Loss:", n_layers, n_single_qubit_params)
    write_log(val_loss, n_layers, n_single_qubit_params)
    write_log("Train Accuracy:", n_layers, n_single_qubit_params)
    write_log(train_acc, n_layers, n_single_qubit_params)
    write_log("Validation Accuracy:", n_layers, n_single_qubit_params)
    write_log(val_acc, n_layers, n_single_qubit_params)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot training and validation loss
    axs[0].plot(train_loss, label="Training Loss", linewidth=2)
    axs[0].plot(val_loss, label="Validation Loss", linewidth=2)
    axs[0].set_title(f"Loss for n_layers={n_layers}, n_single_qubit_params={n_single_qubit_params}", fontsize=16)
    axs[0].set_xlabel("Epochs", fontsize=14)
    axs[0].set_ylabel("Loss", fontsize=14)
    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[0].legend(fontsize=12)

    # Plot training and validation accuracy
    axs[1].plot(train_acc, label="Training Accuracy", linewidth=2)
    axs[1].plot(val_acc, label="Validation Accuracy", linewidth=2)
    axs[1].set_title(f"Accuracy for n_layers={n_layers}, n_single_qubit_params={n_single_qubit_params}", fontsize=16)
    axs[1].set_xlabel("Epochs", fontsize=14)
    axs[1].set_ylabel("Accuracy", fontsize=14)
    axs[1].tick_params(axis='both', which='major', labelsize=12)
    axs[1].legend(fontsize=12)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{n_layers}_{n_single_qubit_params}_loss_accuracy.png")
    plt.close(fig)

def run_experiments_all():
    for n_layers in range(0,5):
        for n_single_qubit_params in range(0,5):
            ansatz=IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=n_layers, n_single_qubit_params=n_single_qubit_params)
            train_circuits = [ansatz(diagram) for diagram in train_diagrams]
            dev_circuits = [ansatz(diagram) for diagram in dev_diagrams]
            test_circuits = [ansatz(diagram) for diagram in test_diagrams]
            all_circuits = train_circuits + dev_circuits + test_circuits

            write_log(f"Training with n_layers: {n_layers} and n_single_qubit_params: {n_single_qubit_params}", n_layers, n_single_qubit_params)
            
            train_loss, val_loss, train_acc, val_acc = train_model(all_circuits, train_circuits, dev_circuits, rewriter, ansatz)

            write_log("Train Loss:", n_layers, n_single_qubit_params)
            write_log(train_loss, n_layers, n_single_qubit_params)
            write_log("Validation Loss:", n_layers, n_single_qubit_params)
            write_log(val_loss, n_layers, n_single_qubit_params)
            write_log("Train Accuracy:", n_layers, n_single_qubit_params)
            write_log(train_acc, n_layers, n_single_qubit_params)
            write_log("Validation Accuracy:", n_layers, n_single_qubit_params)
            write_log(val_acc, n_layers, n_single_qubit_params)

# Main program
if __name__ == '__main__':
    current_time = sys.argv[3]
    log_dir = os.path.join('logs', current_time)
    os.makedirs(log_dir, exist_ok=True)
    plot_dir = os.path.join(log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    train_labels, train_data = read_data('datasets/mc_train_data.txt')
    dev_labels, dev_data = read_data('datasets/mc_dev_data.txt')
    test_labels, test_data = read_data('datasets/mc_test_data.txt')
    parser = BobcatParser(verbose='text')
    raw_train_diagrams = parser.sentences2diagrams(train_data)
    raw_dev_diagrams = parser.sentences2diagrams(dev_data)
    raw_test_diagrams = parser.sentences2diagrams(test_data)
    rewriter = re_norm_cur_norm
    train_diagrams = rewriter(raw_train_diagrams)
    dev_diagrams = rewriter(raw_dev_diagrams)
    test_diagrams = rewriter(raw_test_diagrams)

    shots = 8192
    backend = AerBackend()
    backend_config = {
        'backend': backend,
        'compilation': backend.default_compilation_pass(2),
        'shots': shots
    }

    # run_experiments_all()
    n_layers=int(sys.argv[1])
    n_single_qubit_params=int(sys.argv[2])
    # run_experiment()
    run_specific_experiments()

