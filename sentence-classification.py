
import numpy as np
import warnings
import os
from datasets import load_dataset
from lambeq import BobcatParser
from lambeq import RemoveCupsRewriter
import discopy
from lambeq import Rewriter
from lambeq import AtomicType, IQPAnsatz, MPSAnsatz, Sim14Ansatz, Sim15Ansatz, SpiderAnsatz, StronglyEntanglingAnsatz, Symbol, TensorAnsatz
from lambeq.backend.tensor import Dim
from pytket.extensions.qiskit import AerBackend
from lambeq import TketModel
from lambeq import BinaryCrossEntropyLoss
from lambeq import QuantumTrainer, SPSAOptimizer
from lambeq import Dataset
# print(lambeq.__version__)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def read_data(filename):
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            t = int(line[0])
            labels.append([t, 1-t])
            sentences.append(line[1:].strip())
    return labels, sentences

def diagram_rewriting(diagrams): # (data):
    #diagrams = parser.sentences2diagrams(data)
    rewriter = Rewriter(['prepositional_phrase', 'determiner'])
    rewritten_diagrams = [rewriter(diagram) for diagram in diagrams] 
    normalised_diagrams = [diagram.normal_form() for diagram in rewritten_diagrams]
    curry_functor = Rewriter(['curry'])
    curried_diagrams =[curry_functor(diagram) for diagram in normalised_diagrams] 
    normalised_diagrams2 = [diagram.normal_form() for diagram in curried_diagrams]
    return normalised_diagrams2

train_labels, train_data = read_data('datasets/mc_train_data.txt')
dev_labels, dev_data = read_data('datasets/mc_dev_data.txt')
test_labels, test_data = read_data('datasets/mc_test_data.txt')

parser = BobcatParser(verbose='text', device=0) # device=-1 for CPU

raw_train_diagrams = parser.sentences2diagrams(train_data)
raw_train_diagrams[0].draw()
raw_dev_diagrams = parser.sentences2diagrams(dev_data)
raw_dev_diagrams[0].draw()
raw_test_diagrams = parser.sentences2diagrams(test_data)
raw_test_diagrams[0].draw()

#rewriter = Rewriter(['prepositional_phrase', 'determiner'])
#rewritten_diagram = rewriter(diagram)

# train_diagrams = raw_train_diagrams
# dev_diagrams = raw_dev_diagrams
# test_diagrams = raw_test_diagrams

remove_cups = RemoveCupsRewriter()
train_diagrams = [remove_cups(diagram) for diagram in raw_train_diagrams]
dev_diagrams = [remove_cups(diagram) for diagram in raw_dev_diagrams]
test_diagrams = [remove_cups(diagram) for diagram in raw_test_diagrams]

# train_diagrams = diagram_rewriting(raw_train_diagrams) #[diagram.normal_form() for diagram in raw_train_diagrams]
# dev_diagrams = diagram_rewriting(raw_dev_diagrams) #[diagram.normal_form()for diagram in raw_dev_diagrams]
# test_diagrams = diagram_rewriting(raw_test_diagrams) #[diagram.normal_form() for diagram in raw_test_diagrams]
# train_diagrams[0].draw()

# ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
#                    n_layers=1, n_single_qubit_params=3)
# ansatz = StronglyEntanglingAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, 
#                                   n_layers=2, n_single_qubit_params=3)
ansatz = Sim14Ansatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
                   n_layers=2, n_single_qubit_params=3)
# ansatz = Sim15Ansatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
#                    n_layers=2, n_single_qubit_params=3)

train_circuits = [ansatz(diagram) for diagram in train_diagrams]
train_circuits[0].draw(figsize=(9, 12))
dev_circuits =  [ansatz(diagram) for diagram in dev_diagrams]
test_circuits = [ansatz(diagram) for diagram in test_diagrams]
# test_circuits[0].draw(figsize=(9, 12))

all_circuits = train_circuits+dev_circuits+test_circuits

backend = AerBackend()
backend_config = {
    'backend': backend,
    'compilation': backend.default_compilation_pass(2),
    'shots': 8192
}

model = TketModel.from_diagrams(all_circuits, backend_config=backend_config)
#loss = lambda y_hat, y: -np.sum(y * np.log(y_hat)) / len(y)  # binary cross-entropy loss
loss = BinaryCrossEntropyLoss()
acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2  # half due to double-counting

EPOCHS = 120
BATCH_SIZE = 30

trainer = QuantumTrainer(
    model,
    loss_function=loss,
    epochs=EPOCHS,
    optimizer=SPSAOptimizer,
    optim_hyperparams={'a': 0.05, 'c': 0.06, 'A':0.01*EPOCHS},
    evaluate_functions={'acc': acc},
    evaluate_on_train=True,
    verbose = 'text',
    seed=0
)

train_dataset = Dataset(
            train_circuits,
            train_labels,
            batch_size=BATCH_SIZE)

val_dataset = Dataset(dev_circuits, dev_labels, shuffle=False)
trainer.fit(train_dataset, val_dataset) #, logging_step=1)

# model = TketModel.from_checkpoint('./models/classifier-Sim15-rewrite.pickle', backend_config=backend_config)
model.save("./models/classifier-Sim14-no-cups.pickle")

def measure_score(model):
        test_pred = model.get_diagram_output(test_circuits).tolist()
        total = 0
        for i in range(len(test_pred)):
            if test_pred[i][0]>=0.5:
                test_pred[i][0],test_pred[i][1]=1.0,0.0
            else:
                test_pred[i][0],test_pred[i][1]=0.0,1.0
            if test_pred[i][0]==test_labels[i][0] and test_pred[i][1]==test_labels[i][1]:
                total+=1
        return total/len(test_labels)

print("Model Accuracy: ", measure_score(model))
