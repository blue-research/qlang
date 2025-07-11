import numpy as np
import pandas as pd
import warnings
import os
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
def write_log(log, ansatz_name, rewriter_name):
    print(str(log)) #stdout
    log_path=f"{log_dir}/{ansatz_name}_{rewriter_name}.log"
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
def train_model(all_circuits, train_circuits, dev_circuits, rewriter, ansatz):
    model = TketModel.from_diagrams(all_circuits, backend_config=backend_config)
    loss = BinaryCrossEntropyLoss()
    acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2  # half due to double-counting

    EPOCHS = 10
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
    return model, trainer.train_epoch_costs, trainer.val_costs, trainer.train_eval_results['acc'], trainer.val_eval_results['acc']

def evaluate_model(model, test_circuits, test_labels):
    acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2
    test_pred = model.get_diagram_output(test_circuits).tolist()
    return acc(test_pred, test_labels)

# Experiment and plot saving function
def run_experiment_with_saving_plots():
    for i, ansatz in enumerate(ansatzes):
        ansatz_name = type(ansatz).__name__
        fig, axs = plt.subplots(2, len(rewriters), figsize=(20, 10), sharex=True, sharey=False)

        for j, rewriter in enumerate(rewriters):
            rewriter_name = rewriter.__name__
            train_diagrams = rewriter(raw_train_diagrams)
            dev_diagrams = rewriter(raw_dev_diagrams)
            test_diagrams = rewriter(raw_test_diagrams)

            train_circuits = [ansatz(diagram) for diagram in train_diagrams]
            dev_circuits = [ansatz(diagram) for diagram in dev_diagrams]
            test_circuits = [ansatz(diagram) for diagram in test_diagrams]
            all_circuits = train_circuits + dev_circuits + test_circuits

            # Train model and retrieve losses and accuracies
            write_log(f"Training with {ansatz_name} and {rewriter_name}", ansatz_name, rewriter_name)
            model, train_loss, val_loss, train_acc, val_acc = train_model(all_circuits, train_circuits, dev_circuits, rewriter, ansatz)
            
            # Log results
            write_log("Train Loss: ", ansatz_name, rewriter_name)
            write_log(train_loss, ansatz_name, rewriter_name)
            write_log("Validation Loss: ", ansatz_name, rewriter_name)
            write_log(val_loss, ansatz_name, rewriter_name)
            write_log("Train Accuracy: ", ansatz_name, rewriter_name)
            write_log(train_acc, ansatz_name, rewriter_name)
            write_log("Validation Accuracy: ", ansatz_name, rewriter_name)
            write_log(val_acc, ansatz_name, rewriter_name)
            
            # Plot losses
            axs[0, j].plot(train_loss, label="Training Loss")
            axs[0, j].plot(val_loss, label="Validation Loss")
            axs[0, j].set_title(f"Loss: {ansatz_name} with {rewriter_name}")
            axs[0, j].legend()

            # Plot accuracies
            axs[1, j].plot(train_acc, label="Training Accuracy")
            axs[1, j].plot(val_acc, label="Validation Accuracy")
            axs[1, j].set_title(f"Accuracy: {ansatz_name} with {rewriter_name}")
            axs[1, j].legend()
            
            write_log("Test Accuracy: ", evaluate_model(model, test_circuits, test_labels), ansatz_name, rewriter_name)

        # Save the plot for the current Ansatz with all rewriters as a separate file
        plt.suptitle(f"Loss and Accuracy Curves for {ansatz_name}")
        filename = f"{plot_dir}/{ansatz_name}_loss_accuracy_plots.png"
        plt.savefig(filename)
        plt.close(fig)

# Main program
if __name__ == '__main__':
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('logs', current_time)
    plot_dir = os.path.join(log_dir, "plots")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
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
