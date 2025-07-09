"""app: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from app.task import Net, get_weights


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    
    # Read power-of-choice specific configs
    algo = context.run_config.get("algo", "rand")
    power_of_choice = context.run_config.get("power-of-choice", 2)
    clients_per_round = context.run_config.get("clients-per-round", 10)
    delete_ratio = context.run_config.get("delete-ratio", 0.2)
    rnd_ratio = context.run_config.get("rnd-ratio", 0.3)

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        algo=algo,
        power_of_choice=power_of_choice,
        clients_per_round=clients_per_round,
        delete_ratio=delete_ratio,
        rnd_ratio=rnd_ratio,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
