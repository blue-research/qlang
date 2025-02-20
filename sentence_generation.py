import torch
import random
import numpy as np
import pennylane as qml
from lambeq import AtomicType, BobcatParser, Rewriter, Dataset, IQPAnsatz, MPSAnsatz, TensorAnsatz, PytorchModel, PennyLaneModel, PytorchTrainer
from lambeq.backend.tensor import Dim
from pytket.extensions.qiskit import AerBackend

BATCH_SIZE = 10
EPOCHS = 15
LEARNING_RATE = 0.1
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def read_data(filename):
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            t = float(line[0])
            labels.append([t, 1-t])
            sentences.append(line[1:].strip())
    return labels, sentences

def rewrite(diagrams):
    rewriter = Rewriter(['prepositional_phrase', 'determiner'])
    rewritten_diagrams = [rewriter(diagram) for diagram in diagrams]
    return [diagram.normal_form() for diagram in rewritten_diagrams]

def prepare_circuits(diagrams):
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE
    # ansatz = MPSAnsatz({N: Dim(2), S: Dim(2)}, bond_dim=2)
    ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
                    n_layers=2, n_single_qubit_params=3)
    return [ansatz(d) for d in diagrams]
    # tensor_networks = [ansatz(d).to_tn(torch.float32) for d in diagrams]
    # print(tensor_networks[0])
    # return [torch.tensor(tn, dtype=torch.float32) for tn in tensor_networks]

def acc(y_hat, y):
    return (torch.argmax(y_hat, dim=1) ==
            torch.argmax(y, dim=1)).sum().item()/len(y)

def loss(y_hat, y):
    return torch.nn.functional.mse_loss(y_hat, y)

class QI_VAE(torch.nn.Module):
    def __init__(self, n_qubits=4, latent_dim=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.latent_dim = latent_dim
        
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def encoder(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return qml.probs(wires=range(latent_dim))
        
        @qml.qnode(dev, interface="torch")
        def decoder(inputs, weights): # inputs = latent vector
            qml.AngleEmbedding(inputs, wires=range(latent_dim))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return qml.probs(wires=range(n_qubits))
        
        self.encoder = qml.qnn.TorchLayer(encoder, {"weights": (3, n_qubits)})
        self.decoder = qml.qnn.TorchLayer(decoder, {"weights": (3, n_qubits)})

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent), latent

class HybridModel(torch.nn.Module):
    def __init__(self, qivae, classifier):
        super().__init__()
        self.qivae = qivae
        self.classifier = classifier  # Load pre-trained classifier

    def forward(self, x, target_label):
        reconstructed, latent = self.qivae(x)
        
        # Reconstruction loss
        recon_loss = torch.nn.functional.mse_loss(reconstructed, x)
        
        # Topic loss
        with torch.no_grad():  # Freeze classifier
            topic_probs = self.classifier(x.unsqueeze(0))
        topic_loss = torch.nn.functional.nll_loss(
            topic_probs.log(), torch.tensor([target_label]))
        
        return recon_loss + 0.5 * topic_loss

# 4. Quantum Walk Generation
def quantum_walk_search(model, seed_tensor, target_topic, steps=10):
    model.eval()
    with torch.no_grad():
        latent = model.qivae.encoder(seed_tensor)
        
        for _ in range(steps):
            perturbation = torch.randn_like(latent) * 0.1
            new_latent = (latent + perturbation).view(-1)
            new_latent /= torch.norm(new_latent)
            
            reconstructed = model.qivae.decoder(new_latent)
            if model.classifier(reconstructed.unsqueeze(0)).argmax() == target_topic:
                return reconstructed
    return reconstructed

if __name__ == '__main__':
    train_labels, train_data = read_data('datasets/mc_train_data.txt')
    dev_labels, dev_data = read_data('datasets/mc_dev_data.txt')
    test_labels, test_data = read_data('datasets/mc_test_data.txt')

    parser = BobcatParser(verbose='text')
    train_circuits = parser.sentences2diagrams(train_data)
    dev_circuits = parser.sentences2diagrams(dev_data)
    test_circuits = parser.sentences2diagrams(test_data)

    train_circuits = rewrite(train_circuits)
    dev_circuits = rewrite(dev_circuits)
    test_circuits = rewrite(test_circuits)

    train_circuits = prepare_circuits(train_circuits)
    dev_circuits = prepare_circuits(dev_circuits)
    test_circuits = prepare_circuits(test_circuits)
    all_circuits = train_circuits + dev_circuits + test_circuits
    all_labels = train_labels + dev_labels + test_labels

    backend_config = {'backend': 'default.qubit'}
    model = PennyLaneModel.from_diagrams(all_circuits,
                                        probabilities=True,
                                        normalize=True,
                                        backend_config=backend_config)
    model.initialise_weights()

    train_dataset = Dataset(train_circuits,
                            train_labels,
                            batch_size=BATCH_SIZE)

    val_dataset = Dataset(dev_circuits, dev_labels)

    trainer = PytorchTrainer(
        model=model,
        loss_function=loss,
        optimizer=torch.optim.Adam,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        evaluate_functions={'acc': acc},
        evaluate_on_train=True,
        use_tensorboard=False,
        verbose='text',
        seed=SEED)
    
    trainer.fit(train_dataset, val_dataset)
    
    qivae = QI_VAE()
    classifier = PytorchModel.from_checkpoint('models/pretrained_classifier_tensor.lt')  # Load pretrained model
    hybrid_model = HybridModel(qivae, classifier)
    optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=0.001)
    
    TOPIC_MAP = {"food": 0, "IT": 1}
    target_topic = TOPIC_MAP["food"]
    
    for epoch in range(100):
        total_loss = 0
        for circuit, label in zip(train_circuits, train_labels):
            optimizer.zero_grad()
            loss = hybrid_model(circuit, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Generate example
        seed_tensor = train_circuits[0]
        generated_tensor = quantum_walk_search(hybrid_model, seed_tensor, target_topic)
        
        # Convert back to diagram (for interpretation)
        generated_array = generated_tensor.numpy()
        generated_diagram = TensorAnsatz().interpret(generated_array)
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_circuits):.4f}")
        print("Generated:", generated_diagram.foliation().sentence())
