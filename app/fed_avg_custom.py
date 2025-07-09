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

from .task import Net, get_weights


class FedCustomStrategy(Strategy):
    """Estratégia customizada com algoritmos de seleção pow-d, rpow-d e rand."""
    
    def __init__(
        self,
        algorithm: str = "rand",  # "rand", "pow-d", "rpow-d"
        power_d: int = 10,  # parâmetro d para pow-d e rpow-d
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        client_ratios: Optional[List[float]] = None,  # ratios p_k para cada cliente
    ) -> None:
        super().__init__()
        
        # Parâmetros da estratégia
        self.algorithm = algorithm
        self.power_d = power_d
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        
        # Estado dos clientes
        self.client_losses = {}  # loss atual de cada cliente
        self.client_losses_proxy = {}  # loss anterior (proxy) de cada cliente
        self.client_ratios = client_ratios  # ratios p_k para cada cliente
        self.server_round = 0
        
        # Inicializar ratios uniformes se não fornecido
        if self.client_ratios is None:
            self.client_ratios = []

    def __repr__(self) -> str:
        return f"FedCustomStrategy(algorithm={self.algorithm})"

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
        """Configurar próxima rodada de treinamento com seleção de clientes."""
        
        self.server_round = server_round
        num_available_clients = client_manager.num_available()
        
        # Inicializar ratios uniformes se necessário
        if len(self.client_ratios) != num_available_clients:
            self.client_ratios = [1.0 / num_available_clients] * num_available_clients
        
        # Calcular número de clientes para seleção
        clients_per_round = max(
            int(num_available_clients * self.fraction_fit),
            self.min_fit_clients
        )
        
        # Obter lista de todos os clientes disponíveis
        all_clients = list(client_manager.all().values())
        
        # Selecionar clientes baseado no algoritmo escolhido
        selected_client_indices = self._select_clients(
            num_available_clients=num_available_clients,
            clients_per_round=clients_per_round,
            server_round=server_round
        )
        
        # Criar configurações para os clientes selecionados
        config = {"server_round": server_round}
        fit_configurations = []
        
        for client_idx in selected_client_indices:
            if client_idx < len(all_clients):
                client = all_clients[client_idx]
                fit_configurations.append((client, FitIns(parameters, config)))
        
        return fit_configurations

    def _select_clients(
        self, 
        num_available_clients: int, 
        clients_per_round: int, 
        server_round: int
    ) -> List[int]:
        """
        Selecionar clientes baseado no algoritmo escolhido.
        
        Returns:
            Lista de índices dos clientes selecionados
        """
        
        if self.algorithm == "rand":
            selected_indices = np.random.choice(
                num_available_clients, 
                p=self.client_ratios, 
                size=clients_per_round, 
                replace=True
            )
            return selected_indices.tolist()
        elif self.algorithm == "pow-d":
            return self._power_of_choice_selection(
                num_available_clients, 
                clients_per_round, 
                use_proxy=False
            )
            
        elif self.algorithm == "rpow-d":
            return self._power_of_choice_selection(
                num_available_clients, 
                clients_per_round, 
                use_proxy=True
            )

    def _power_of_choice_selection(
        self, 
        num_available_clients: int, 
        clients_per_round: int, 
        use_proxy: bool = False
    ) -> List[int]:
        """
        Implementar algoritmo power-of-choice (pow-d ou rpow-d).
        
        Args:
            num_available_clients: Número total de clientes disponíveis
            clients_per_round: Número de clientes para selecionar (m)
            use_proxy: Se True, usa rpow-d (proxy loss), senão pow-d (loss atual)
        """
        
        # Step 1: Selecionar 'd' clientes com probabilidade proporcional ao dataset size
        d_candidates = min(self.power_d, num_available_clients)
        candidate_indices = np.random.choice(
            num_available_clients, 
            p=self.client_ratios, 
            size=d_candidates, 
            replace=False
        )
        
        # Step 2: Obter losses dos candidatos
        candidate_losses = []
        for client_idx in candidate_indices:
            if use_proxy:
                # rpow-d: usar loss anterior (proxy)
                loss = self.client_losses_proxy.get(client_idx, 0.0)
            else:
                # pow-d: usar loss atual
                loss = self.client_losses.get(client_idx, 0.0)
            candidate_losses.append(loss)
        
        # Step 3: Ordenar candidatos por loss decrescente e selecionar top m
        candidates_with_losses = list(zip(candidate_losses, candidate_indices))
        candidates_with_losses.sort(key=lambda x: x[0], reverse=True)
        
        # Selecionar os top m clientes
        m_clients = min(clients_per_round, len(candidates_with_losses))
        selected_indices = [
            client_idx for _, client_idx in candidates_with_losses[:m_clients]
        ]
        
        return selected_indices

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Agregar resultados do treinamento usando média ponderada."""
        
        if not results:
            return None, {}
        
        # Atualizar losses dos clientes para próximas seleções
        self._update_client_losses(results)
        
        # Agregar parâmetros usando média ponderada
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        
        # Métricas agregadas
        metrics_aggregated = {
            "num_selected_clients": len(results),
            "algorithm": self.algorithm,
        }
        
        return parameters_aggregated, metrics_aggregated

    def _update_client_losses(self, results: List[Tuple[ClientProxy, FitRes]]):
        """Atualizar histórico de losses dos clientes."""
        
        # Mover losses atuais para proxy (para rpow-d)
        self.client_losses_proxy = self.client_losses.copy()
        
        # Atualizar com novos losses
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            
            # Tentar extrair loss de diferentes fontes
            loss_value = None
            
            # 1. Tentar obter loss das métricas
            if hasattr(fit_res, 'metrics') and fit_res.metrics:
                if 'train_loss' in fit_res.metrics:
                    loss_value = fit_res.metrics['train_loss']
                elif 'loss' in fit_res.metrics:
                    loss_value = fit_res.metrics['loss']
            
            # 2. Se não encontrar loss, usar heurística baseada no número de exemplos
            if loss_value is None:
                # Clientes com mais dados tendem a ter loss menor (heurística)
                # Normalizar pelo número de exemplos para criar diversidade
                total_examples = sum(res.num_examples for _, res in results)
                client_proportion = fit_res.num_examples / total_examples
                # Inverter proporção para que clientes menores tenham "loss" maior
                loss_value = 1.0 - client_proportion + np.random.normal(0, 0.1)
            
            # Converter client_id para int se necessário
            try:
                client_idx = int(client_id)
                self.client_losses[client_idx] = float(loss_value)
            except (ValueError, TypeError):
                # Se não conseguir converter, usar hash do client_id
                client_idx = hash(client_id) % 10000
                self.client_losses[client_idx] = float(loss_value)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configurar próxima rodada de avaliação."""
        
        if self.fraction_evaluate == 0.0:
            return []
        
        config = {"server_round": server_round}
        evaluate_ins = EvaluateIns(parameters, config)
        
        # Selecionar clientes para avaliação
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Agregar resultados da avaliação usando média ponderada."""
        
        if not results:
            return None, {}
        
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        
        # Calcular métricas agregadas
        metrics_aggregated = {}
        if results:
            # Agregar accuracy se disponível
            accuracies = [
                (evaluate_res.num_examples, evaluate_res.metrics.get("accuracy", 0.0))
                for _, evaluate_res in results
                if hasattr(evaluate_res, 'metrics') and 'accuracy' in evaluate_res.metrics
            ]
            if accuracies:
                total_examples = sum(num_examples for num_examples, _ in accuracies)
                weighted_accuracy = sum(
                    num_examples * accuracy for num_examples, accuracy in accuracies
                ) / total_examples
                metrics_aggregated["accuracy"] = weighted_accuracy
        
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Avaliar parâmetros globais do modelo (opcional)."""
        # Não implementar avaliação centralizada no servidor
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Retornar tamanho da amostra e número mínimo de clientes necessários."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Usar fração dos clientes disponíveis para avaliação."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients