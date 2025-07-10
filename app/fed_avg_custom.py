from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from app.task import Net, get_weights

class FedCustomStrategy(Strategy):
    """
    Estratégia customizada com amostragem ponderada pelo tamanho do dataset para
    os algoritmos pow-d e rpow-d.
    """
    
    def __init__(
        self,
        algorithm: str = "rand",
        power_d: int = 10,
        fraction_fit: float = 0.1,
        fraction_evaluate: float = 0.1,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ) -> None:
        super().__init__()
        
        if algorithm not in ["rand", "pow-d", "rpow-d"]:
            raise ValueError(f"Algoritmo desconhecido: {algorithm}")
            
        self.algorithm = algorithm
        self.power_d = power_d
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        
        self.client_losses: Dict[str, float] = {}
        self.client_losses_proxy: Dict[str, float] = {}
        self.client_num_examples: Dict[str, int] = {}

    def __repr__(self) -> str:
        return f"FedCustomStrategy(algorithm={self.algorithm}, d={self.power_d})"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Inicializar parâmetros globais do modelo."""
        net = Net()
        ndarrays = get_weights(net)
        return ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configurar a próxima rodada de treinamento."""
        num_available_clients = client_manager.num_available()
        if num_available_clients < self.min_available_clients:
            return []

        num_clients_to_select = int(self.fraction_fit * num_available_clients)
        num_clients_to_select = max(num_clients_to_select, self.min_fit_clients)
        
        all_clients = list(client_manager.all().values())
        
        selected_clients = self._select_clients(all_clients, num_clients_to_select)
        
        config = {"server_round": server_round}
        fit_ins = FitIns(parameters, config)
        
        return [(client, fit_ins) for client in selected_clients]

    def _select_clients(
        self, 
        all_clients: List[ClientProxy], 
        num_clients_to_select: int,
    ) -> List[ClientProxy]:
        """Seleciona clientes com base no algoritmo."""
        if self.algorithm == "rand":
            return np.random.choice(all_clients, num_clients_to_select, replace=False).tolist()
        
        elif self.algorithm in ["pow-d", "rpow-d"]:
            return self._power_of_choice_selection(
                all_clients, 
                num_clients_to_select,
                use_proxy=(self.algorithm == "rpow-d")
            )
        return []

    def _power_of_choice_selection(
        self, 
        all_clients: List[ClientProxy], 
        num_clients_to_select: int, 
        use_proxy: bool
    ) -> List[ClientProxy]:
        """Implementa Power-of-Choice com amostragem ponderada."""
        num_available = len(all_clients)
        num_candidates = min(self.power_d, num_available)

        all_cids = [client.cid for client in all_clients]
        weights = np.array([self.client_num_examples.get(cid, 1) for cid in all_cids])
        
        # Normaliza os pesos para obter as probabilidades
        sum_weights = np.sum(weights)
        if sum_weights == 0:
            # Fallback para amostragem uniforme se todos os pesos forem zero (ex: primeira rodada)
            probabilities = None
        else:
            probabilities = weights / sum_weights

        # Etapa 1: Selecionar 'd' candidatos com base nas probabilidades p_k
        candidate_indices = np.random.choice(
            num_available, 
            size=num_candidates, 
            replace=False,
            p=probabilities # Usa as probabilidades calculadas
        )
        candidate_proxies = [all_clients[i] for i in candidate_indices]
        
        # Etapa 2: Obter perdas dos candidatos
        loss_source = self.client_losses_proxy if use_proxy else self.client_losses
        candidate_losses = [loss_source.get(p.cid, float('inf')) for p in candidate_proxies]
        
        # Etapa 3: Ordenar e selecionar os 'm' melhores
        candidates_with_losses = sorted(
            zip(candidate_losses, candidate_proxies), 
            key=lambda x: x[0], 
            reverse=True
        )
        
        num_to_return = min(num_clients_to_select, len(candidates_with_losses))
        return [proxy for _, proxy in candidates_with_losses[:num_to_return]]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Agregar resultados e atualizar estado dos clientes."""
        if not results:
            return None, {}
        
        self._update_client_state(results)
        
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        
        return parameters_aggregated, {}

    def _update_client_state(self, results: List[Tuple[ClientProxy, FitRes]]):
        """Atualiza perdas e contagem de exemplos dos clientes."""
        self.client_losses_proxy = self.client_losses.copy()
        
        for client_proxy, fit_res in results:
            # Atualiza o número de exemplos para a amostragem ponderada
            self.client_num_examples[client_proxy.cid] = fit_res.num_examples
            
            # Atualiza a perda do cliente
            if fit_res.metrics and 'train_loss' in fit_res.metrics:
                self.client_losses[client_proxy.cid] = float(fit_res.metrics['train_loss'])

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configurar rodada de avaliação."""
        if self.fraction_evaluate == 0.0:
            return []
        
        config = {"server_round": server_round}
        evaluate_ins = EvaluateIns(parameters, config)
        
        sample_size = int(client_manager.num_available() * self.fraction_evaluate)
        clients = client_manager.sample(
            num_clients=max(sample_size, self.min_evaluate_clients),
            min_num_clients=self.min_evaluate_clients
        )
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Agregar resultados da avaliação."""
        if not results:
            return None, {}
        
        loss_aggregated = weighted_loss_avg([(res.num_examples, res.loss) for _, res in results])
        
        accuracies = [res.metrics["accuracy"] * res.num_examples for _, res in results if "accuracy" in res.metrics]
        examples = [res.num_examples for _, res in results if "accuracy" in res.metrics]

        metrics_aggregated = {}
        if examples:
            metrics_aggregated["accuracy"] = sum(accuracies) / sum(examples)
        
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Avaliação centralizada no servidor (não implementada)."""
        return None