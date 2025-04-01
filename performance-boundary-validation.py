import time
start_time = time.time()


import math
import random
import sys
from kan import *
from kan.utils import create_dataset
from kan.utils import ex_round
torch.set_default_dtype(torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Parameters
input_size = 1
hidden_layers = 5#10
output_size = 1
epochs = 50
learning_rate = 0.001
variance = 0.02
mean = 0
train_samples = 1000
test_samples = 1000
interval = (0, 2*torch.pi)

# Number of tests
iterations = 1000

f = lambda x: torch.sin(x[:,[0]]) # + x[:,[0]]**2
# Use torch.linspace for direct PyTorch tensor creation
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
#lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin']

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
    #noise = torch.normal(mean=mean, std=torch.sqrt(torch.tensor(variance)), size=x_train.size())
    
    # Add noise to the original training data
    noisy_data = x_train + noise

    return noisy_data

def iteration_callback():

    # Use torch.linspace for direct PyTorch tensor creation
    y = add_white_noise(y_clear, 0.0, variance)
    dataset['train_input'] = x
    dataset['train_label'] = y

    # create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
    kan_model = MultKAN(
        width=[input_size, hidden_layers, hidden_layers, output_size],
        grid=5, 
        k=3,
        #seed=42, 
        device=device, 
        #grid_range= [interval[0], interval[1]],
        #noise_scale=0.0,
        #affine_trainable=True,
        #grid_eps=0,
        auto_save=False,
    )

    kan_model.fit(
        dataset, 
        opt = "LBFGS", 
        steps = epochs,
        update_grid = True,
        lamb = learning_rate,
        #lamb_l1 = 100,
        #lamb_entropy = 100,
        #lamb_coef = 100,
        #lamb_coefdiff = 100,
        loss_fn = torch.nn.MSELoss(),
        log=-1,
    )

    kan_model = kan_model.prune(
        node_th = 1e-1
        #node_th = 0.02,
        #edge_th = 0.06
    )
    kan_model.fit(
        dataset, 
        opt = "LBFGS", 
        steps = epochs,
        update_grid = True,
        lamb = learning_rate/2,
        #lamb_l1 = 100,
        #lamb_entropy = 100,
        #lamb_coef = 100,
        #lamb_coefdiff = 100,
        loss_fn = torch.nn.MSELoss(),
        log=-1,
    )

    kan_model.auto_symbolic(
        lib=lib,
        #a_range=(-1,1),
        #b_range=(-1,1),
        r2_threshold=0.9,
        verbose=0,
    )
    kan_model.fit(
        dataset, 
        opt = "LBFGS", 
        steps = epochs,
        update_grid = True,
        lamb = learning_rate,
        #lamb_l1 = 100,
        #lamb_entropy = 100,
        #lamb_coef = 100,
        #lamb_coefdiff = 100,
        loss_fn = torch.nn.MSELoss(),
        log=-1,
    )

    training_data_codomain = (float(min(y)[0]), float(max(y)[0]))
    kan_model_output = kan_model(dataset['train_input']).detach().cpu().numpy()
    kan_model_codomain = (float(min(kan_model_output)[0]), float(max(kan_model_output)[0]))
    equation = ""#ex_round(kan_model.symbolic_formula()[0][0], 4)

    return training_data_codomain, kan_model_codomain, equation



print(f"| Iteration | Data Codomain | KAN Codomain | KAN Equation |")
print(f"|-----------|---------------|--------------|--------------|")
i = 0
while i < iterations:
    try:
        training_data_codomain, kan_model_codomain, equation = iteration_callback()
        output = f"| {i+1} | {training_data_codomain} | {kan_model_codomain} | {equation} |"
        if not ("nan" in output):
            print(output)
            i += 1
    except:
        pass



from IPython.display import display
def parse_table(text):
    lines = text.strip().split("\n")
    data = []
    
    for line in lines[2:]:  # Skip the header lines
        match = re.match(r'\|\s*(\d+)\s*\|\s*\(([^,]+), ([^\)]+)\)\s*\|\s*\(([^,]+), ([^\)]+)\)\s*\|', line)
        if match:
            iteration = int(match.group(1))
            data_codomain = (float(match.group(2)), float(match.group(3)))
            kan_codomain = (float(match.group(4)), float(match.group(5)))
            
            data.append({
                "Iteration": iteration,
                "Data Codomain": data_codomain,
                "KAN Codomain": kan_codomain
            })
    
    return data

def performance_statistics(parsed_data, key, mean, variance, real_codomain=(-1, 1)):
    codomains = []
    deviations = []
    for entry in parsed_data:
        x, y = entry[key]
        codomains.append((x, y))
        deviation_x = abs(x - real_codomain[0])
        deviation_y = abs(y - real_codomain[1])
        deviations.append((deviation_x, deviation_y))
    
    codomains = np.array(codomains)
    deviations = np.array(deviations)
    mean_codomain = np.mean(codomains, axis=0)
    mean_deviation = np.mean(deviations, axis=0)
    max_deviation = np.max(deviations, axis=0)
    min_deviation = np.min(deviations, axis=0)

    below_variance_x = np.sum(deviations[:, 0] <= abs(real_codomain[0]*variance)) / len(deviations) * 100
    below_variance_y = np.sum(deviations[:, 1] <= abs(real_codomain[1]*variance)) / len(deviations) * 100
    
    stats = {
        "Real Codomain": real_codomain,
        "Mean Codomain": mean_codomain,
        "Mean Deviation": mean_deviation,
        "Max Deviation": max_deviation,
        "Min Deviation": min_deviation,
        "Bounding Variance": [real_codomain[0]*variance, real_codomain[1]*variance],
        "% Below Variance": [below_variance_x, below_variance_y]
    }

    df = pd.DataFrame(stats, index=["X", "Y"])
    display(df)
    return df


bounding_variance = 2*variance
parsed_data = parse_table(text)
print("Data Statistics:")
data_statistics = performance_statistics(parsed_data, "Data Codomain", mean, bounding_variance, (-1, 1))

print("KAN Statistics:")
kan_statistics = performance_statistics(parsed_data, "KAN Codomain", mean, bounding_variance, (-1, 1))






finish_time = time.time()
time_in_secs = finish_time - start_time
print(f"Elapsed Time: {time_in_secs} seconds")
