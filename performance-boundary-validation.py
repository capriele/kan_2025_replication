import time
import math
import random
import sys
import torch
import torch.multiprocessing as mp
from kan import *
from kan.utils import create_dataset
from kan.utils import ex_round

torch.set_default_dtype(torch.float32)

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
train_samples = 1000
test_samples = 1000
interval = (0, 2 * torch.pi)

# Number of tests
iterations = 1000
num_processes = 4  # Number of parallel processes

f = lambda x: torch.sin(x[:, [0]])
x = torch.linspace(interval[0], interval[1], train_samples)
x = torch.stack([x.ravel()], dim=-1)
y_clear = f(x)
dataset = create_dataset(
    f,
    n_var=1,
    device=device,
    normalize_input=False,
    normalize_label=False,
    train_num=train_samples,
    test_num=test_samples,
    ranges=[interval[0], interval[1]],
)
lib = ["x", "x^2", "x^3", "x^4", "exp", "log", "sqrt", "tanh", "sin"]


def add_white_noise(x_train, mean=0.0, variance=0.01):
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
    noise = torch.randn(x_train.size()) * torch.sqrt(torch.tensor(variance)) + mean
    noisy_data = x_train + noise
    return noisy_data


def iteration_callback(iteration):
    try:
        y = add_white_noise(y_clear, 0.0, variance)
        dataset["train_input"] = x
        dataset["train_label"] = y
        dataset["train_input"] = dataset["train_input"].to(device)
        dataset["train_label"] = dataset["train_label"].to(device)

        kan_model = MultKAN(
            width=[input_size, hidden_layers, hidden_layers, output_size],
            grid=5,
            k=3,
            device=device,
            auto_save=False,
        )

        kan_model.fit(
            dataset,
            opt="LBFGS",
            steps=epochs,
            update_grid=True,
            lamb=learning_rate,
            loss_fn=torch.nn.MSELoss(),
            log=-1,
        )

        kan_model = kan_model.prune(node_th=1e-1)
        kan_model.fit(
            dataset,
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
            dataset,
            opt="LBFGS",
            steps=epochs,
            update_grid=True,
            lamb=learning_rate,
            loss_fn=torch.nn.MSELoss(),
            log=-1,
        )

        training_data_codomain = (float(min(y)[0]), float(max(y)[0]))
        kan_model_output = kan_model(dataset["train_input"]).detach().cpu().numpy()
        kan_model_codomain = (
            float(min(kan_model_output)[0]),
            float(max(kan_model_output)[0]),
        )
        equation = ""  # ex_round(kan_model.symbolic_formula()[0][0], 4)

        output = f"| {iteration + 1} | {training_data_codomain} | {kan_model_codomain} | {equation} |"
        print(output)
        return output
    except Exception as e:
        print(f"Iteration {iteration + 1} failed: {e}")
        return None


def worker(iteration_range, results):
    for iteration in iteration_range:
        result = iteration_callback(iteration)
        if result:
            results.append(result)


if __name__ == "__main__":
    start_time = time.time()

    print(f"| Iteration | Data Codomain | KAN Codomain | KAN Equation |")
    print(f"|-----------|---------------|--------------|--------------|")

    manager = mp.Manager()
    results = manager.list()

    # Split iterations among processes
    iterations_per_process = iterations // num_processes
    processes = []
    for i in range(num_processes):
        start = i * iterations_per_process
        end = (i + 1) * iterations_per_process if i != num_processes - 1 else iterations
        p = mp.Process(target=worker, args=(range(start, end), results))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Combine results
    table = "| Iteration | Data Codomain | KAN Codomain | KAN Equation |\n"
    table += "|-----------|---------------|--------------|--------------|\n"
    for result in results:
        table += result + "\n"

    output_file = "performance_boundary_validation.txt"
    with open(output_file, "w") as f:
        f.write(table)

    finish_time = time.time()
    time_in_secs = finish_time - start_time
    print(f"Elapsed Time: {time_in_secs} seconds")
