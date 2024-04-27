
import numpy as np
import warnings
import os
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

from lambeq import Rewriter
def diagram_rewritting(diagrams): #(data):
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

import discopy
#print(lambeq.__version__)

from lambeq import BobcatParser

parser = BobcatParser(verbose='text')

raw_train_diagrams = parser.sentences2diagrams(train_data)
raw_train_diagrams[0].draw()
raw_dev_diagrams = parser.sentences2diagrams(dev_data)
raw_dev_diagrams[0].draw()
raw_test_diagrams = parser.sentences2diagrams(test_data)
raw_test_diagrams[0].draw()

#from lambeq import remove_cups
#from lambeq import Rewriter
#rewriter = Rewriter(['prepositional_phrase', 'determiner'])
#rewritten_diagram = rewriter(diagram)

print("Before removing cups")

raw_train_diagrams[0].draw()
# train_diagrams = [remove_cups(diagram) for diagram in raw_train_diagrams]
# print("After removing cups1")
# train_diagrams[0].draw()

train_diagrams = diagram_rewritting(raw_train_diagrams) #[diagram.normal_form() for diagram in raw_train_diagrams]
print("After removing cups2")
train_diagrams[0].draw()


dev_diagrams = diagram_rewritting(raw_dev_diagrams) #[diagram.normal_form()for diagram in raw_dev_diagrams]

dev_diagrams[0].draw()

test_diagrams = diagram_rewritting(raw_test_diagrams) #[diagram.normal_form() for diagram in raw_test_diagrams]
test_diagrams[0].draw()


from lambeq import AtomicType, IQPAnsatz

ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
                   n_layers=1, n_single_qubit_params=3)

train_circuits = [ansatz(diagram) for diagram in train_diagrams]
train_circuits[0].draw(figsize=(9, 12))
dev_circuits =  [ansatz(diagram) for diagram in dev_diagrams]

test_circuits = [ansatz(diagram) for diagram in test_diagrams]
test_circuits[0].draw(figsize=(9, 12))

from pytket.extensions.qiskit import AerBackend
from lambeq import TketModel

all_circuits = train_circuits+dev_circuits+test_circuits

backend = AerBackend()
backend_config = {
    'backend': backend,
    'compilation': backend.default_compilation_pass(2),
    'shots': 8192
}

model = TketModel.from_diagrams(all_circuits, backend_config=backend_config)

#loss = lambda y_hat, y: -np.sum(y * np.log(y_hat)) / len(y)  # binary cross-entropy loss
from lambeq import BinaryCrossEntropyLoss
loss = BinaryCrossEntropyLoss()

acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2  # half due to double-counting

from lambeq import QuantumTrainer, SPSAOptimizer

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

from lambeq import Dataset

train_dataset = Dataset(
            train_circuits,
            train_labels,
            batch_size=BATCH_SIZE)

val_dataset = Dataset(dev_circuits, dev_labels, shuffle=False)
trainer.fit(train_dataset, val_dataset) #, logging_step=1)

model = TketModel.from_checkpoint('classifier.pickle', backend_config=backend_config)
model.save("classifier.pickle")


