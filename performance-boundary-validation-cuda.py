import time
import torch
from torch.cuda import Stream
from kan import *
from kan.utils import create_dataset
import sys

torch.set_default_dtype(torch.float64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Parameters
input_size = 1
hidden_layers = 5
output_size = 1
epochs = 50
learning_rate = 0.001
variance = 0.02
mean = 0
train_samples = 10000
test_samples = 1000
interval = (0, 2 * torch.pi)

# Number of tests
iterations = 1000  # Lower this unless you have multiple GPUs or a beefy one
parallel_streams = 1  # Number of CUDA streams to run in parallel

f = lambda x: torch.sin(x[:, [0]])
x = torch.linspace(interval[0], interval[1], train_samples).unsqueeze(1).to(device)
y_clear = f(x).to(device)
lib = ["x", "x^2", "x^3", "x^4", "exp", "log", "sqrt", "tanh", "sin"]


def add_white_noise(y, mean=0.0, variance=0.01):
    """
    Adds white noise to the input data.

    Args:
        x_train (torch.Tensor): The input training data (e.g., x_train).
        mean (float): Mean of the Gaussian noise to be added.
        variance (float): Variance of the Gaussian noise to be added.

    Returns:
        torch.Tensor: The noisy data.
    """
    # Generate Gaussian noise (mean, variance) with the same shape as x_train
    random.seed(None)
    random_seed = random.randint(-sys.maxsize, sys.maxsize)
    torch.manual_seed(random_seed)
    noise = torch.randn_like(y, device=device) * torch.sqrt(torch.tensor(variance, device=y.device)) + mean
    return y + noise


def iteration_callback(iteration, stream):
    try:
        # Generate noise and data on GPU
        y = add_white_noise(y_clear, 0.0, variance)
        y_test = add_white_noise(y_clear, 0.0, variance)

        # Build dataset, all on GPU
        local_dataset = {
            "train_input": x,
            "train_label": y,
            "test_input": x,
            "test_label": y_test,
        }

        # Initialize model on GPU
        kan_model = MultKAN(
            width=[input_size, hidden_layers, output_size],
            grid=5,
            k=3,
            device=device,
            auto_save=False,
            symbolic_enabled=False,
        )

        # First training phase
        kan_model.fit(
            local_dataset,
            opt="LBFGS",
            steps=epochs,
            update_grid=True,
            lamb=learning_rate,
            loss_fn=torch.nn.MSELoss(),
            log=-1,
        )

        # Prune and refine
        kan_model = kan_model.prune(node_th=1e-1)
        kan_model.fit(
            local_dataset,
            opt="LBFGS",
            steps=epochs,
            update_grid=True,
            lamb=learning_rate / 2,
            loss_fn=torch.nn.MSELoss(),
            log=-1,
        )

        kan_model.auto_symbolic(
            lib=lib,
            r2_threshold=0.9,
            verbose=0,
        )
        kan_model.fit(
            local_dataset,
            opt="LBFGS",
            steps=epochs,
            update_grid=True,
            lamb=learning_rate,
            loss_fn=torch.nn.MSELoss(),
            log=-1,
        )

        # Evaluate model, all in GPU
        with torch.no_grad():
            model_output = kan_model(local_dataset["train_input"])

            training_data_codomain = (
                torch.min(y).item(),
                torch.max(y).item()
            )
            model_output_codomain = (
                torch.min(model_output).item(),
                torch.max(model_output).item()
            )
            equation = ""  # Still GPU-only, unless symbolic conversion is needed

        output = (
            f"| {iteration + 1} "
            f"| {training_data_codomain} "
            f"| {model_output_codomain} "
            f"| {equation} |"
        )
        #print(output)
        return output
    except Exception as e:
        #print(f"Iteration {iteration + 1} failed: {e}")
        return None

if __name__ == "__main__":
    start_time = time.time()
    print(f"| Iteration | Data Codomain | KAN Codomain | KAN Equation |")
    print(f"|-----------|---------------|--------------|--------------|")

    streams = [Stream(device=device) for _ in range(parallel_streams)]
    results = [None] * iterations
    torch.cuda.synchronize()

    pending = []
    i = 0
    while i < iterations:
        for j in range(parallel_streams):
            with torch.cuda.stream(streams[j]):
                results[i] = iteration_callback(i, streams[j])
            print(results[i])
            i += 1
    table = "| Iteration | Data Codomain | KAN Codomain | KAN Equation |\n"
    table += "|-----------|---------------|--------------|--------------|\n"
    for result in results:
        if result:
            table += result + "\n"

    output_file = "performance_boundary_validation_stream.txt"
    with open(output_file, "w") as f:
        f.write(table)

    elapsed_time = time.time() - start_time
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
