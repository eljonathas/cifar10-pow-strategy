"""app: A Flower / PyTorch app."""

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from app.fed_avg_custom import FedCustomStrategy


def server_fn(context: Context):
    fraction_fit = context.run_config["fraction-fit"]
    num_rounds = context.run_config["num-server-rounds"]

    strategy = FedCustomStrategy(
        algorithm="rpow-d",
        power_d=10,
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
