import numpy as np
import random
from typing import List, Tuple, Optional, Dict, Any
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, FitRes, EvaluateRes
from flwr.server.client_manager import ClientManager


class FedAvgPowerOfChoice(FedAvg):
    def __init__(
        self,
        algo: str = "rand",
        power_of_choice: int = 2,
        clients_per_round: int = 10,
        delete_ratio: float = 0.2,
        rnd_ratio: float = 0.3,
        **kwargs
    ):
        """
        Estratégia FedAvg com seleção de clientes baseada em power-of-choice.
        
        Args:
            algo: Algoritmo de seleção ('rand', 'pow-d', 'cpow-d', 'adapow-d', 'rpow-d', 
                  'pow-dint', 'rpow-dint', 'afl', 'pow-norm', 'randint')
            power_of_choice: Parâmetro d para estratégias power-of-choice
            clients_per_round: Número de clientes por rodada (m)
            delete_ratio: Proporção de clientes a serem excluídos (para afl)
            rnd_ratio: Proporção de seleção aleatória (para afl)
        """
        super().__init__(**kwargs)
        self.algo = algo
        self.power_of_choice = power_of_choice
        self.clients_per_round = clients_per_round
        self.delete_ratio = delete_ratio
        self.rnd_ratio = rnd_ratio
        
        # Métricas de clientes (serão populadas durante a execução)
        self.client_losses: Dict[str, float] = {}
        self.client_loss_proxy: Dict[str, float] = {}
        self.client_ratios: Dict[str, float] = {}
        self.current_round = 0

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, Dict[str, Any]]]:
        """Configure a próxima rodada de treinamento."""
        self.current_round = server_round
        
        # Obter todos os clientes disponíveis
        all_clients = list(client_manager.all().values())
        
        # Inicializar ratios se não existirem
        if not self.client_ratios:
            self._initialize_client_ratios(all_clients)
        
        # Selecionar clientes baseado no algoritmo
        selected_clients = self._select_clients(all_clients)
        
        # Configurar parâmetros para os clientes selecionados
        config = {"server_round": server_round}
        client_instructions = []
        
        for client in selected_clients:
            client_instructions.append((client, config))
        
        return client_instructions

    def _initialize_client_ratios(self, clients: List[ClientProxy]):
        """Inicializa os ratios dos clientes (assumindo distribuição uniforme inicialmente)."""
        num_clients = len(clients)
        for client in clients:
            self.client_ratios[client.cid] = 1.0 / num_clients

    def _select_clients(self, available_clients: List[ClientProxy]) -> List[ClientProxy]:
        """
        Seleciona clientes baseado no algoritmo especificado.
        
        Args:
            available_clients: Lista de clientes disponíveis
            
        Returns:
            Lista de clientes selecionados
        """
        num_clients = len(available_clients)
        client_ids = [client.cid for client in available_clients]
        
        # Primeira rodada: seleção aleatória
        if self.current_round == 1 or not self.client_losses:
            selected_indices = np.random.choice(
                num_clients, 
                size=min(self.clients_per_round, num_clients), 
                replace=False
            )
            return [available_clients[i] for i in selected_indices]
        
        # Criar arrays para facilitar o processamento
        client_ratios = np.array([self.client_ratios.get(cid, 1.0/num_clients) for cid in client_ids])
        client_losses = np.array([self.client_losses.get(cid, 0.0) for cid in client_ids])
        client_loss_proxy = np.array([self.client_loss_proxy.get(cid, 0.0) for cid in client_ids])
        
        if self.algo == 'rand':
            # Seleção aleatória com probabilidade proporcional ao tamanho do dataset
            selected_indices = np.random.choice(
                num_clients, 
                p=client_ratios, 
                size=self.clients_per_round, 
                replace=True
            )
            
        elif self.algo == 'randint':
            # Seleção aleatória com disponibilidade intermitente
            selected_indices = self._randint_selection(num_clients, client_ratios)
            
        elif self.algo in ['pow-d', 'cpow-d', 'adapow-d']:
            # Estratégias power-of-choice padrão
            selected_indices = self._power_of_choice_selection(
                num_clients, client_ratios, client_losses
            )
            
        elif self.algo == 'rpow-d':
            # Power-of-choice com proxy de loss
            selected_indices = self._power_of_choice_selection(
                num_clients, client_ratios, client_loss_proxy
            )
            
        elif self.algo == 'pow-dint':
            # Power-of-choice com disponibilidade intermitente
            selected_indices = self._power_of_choice_intermittent(
                num_clients, client_ratios, client_losses
            )
            
        elif self.algo == 'rpow-dint':
            # Power-of-choice com proxy e disponibilidade intermitente
            selected_indices = self._power_of_choice_intermittent(
                num_clients, client_ratios, client_loss_proxy
            )
            
        elif self.algo == 'afl':
            # Estratégia AFL (Active Federated Learning)
            selected_indices = self._afl_selection(num_clients, client_loss_proxy)
            
        else:
            # Fallback para seleção aleatória
            selected_indices = np.random.choice(
                num_clients, 
                size=min(self.clients_per_round, num_clients), 
                replace=False
            )
        
        return [available_clients[i] for i in selected_indices]

    def _power_of_choice_selection(
        self, num_clients: int, client_ratios: np.ndarray, losses: np.ndarray
    ) -> np.ndarray:
        """Implementa a estratégia power-of-choice padrão."""
        # Passo 1: Selecionar d clientes com probabilidade proporcional ao tamanho do dataset
        d = min(self.power_of_choice, num_clients)
        random_indices = np.random.choice(
            num_clients, p=client_ratios, size=d, replace=False
        )
        
        # Passo 2: Ordenar por loss em ordem decrescente
        random_losses = [(losses[i], i) for i in random_indices]
        random_losses.sort(key=lambda x: x[0], reverse=True)
        
        # Passo 3: Selecionar os top m clientes
        m = min(self.clients_per_round, len(random_losses))
        selected_indices = [idx for _, idx in random_losses[:m]]
        
        return np.array(selected_indices)

    def _randint_selection(self, num_clients: int, client_ratios: np.ndarray) -> np.ndarray:
        """Implementa seleção aleatória com disponibilidade intermitente."""
        delete_count = int(self.delete_ratio * num_clients / 2)
        
        if (self.current_round % 2) == 0:
            # Rodadas pares: remover clientes da primeira metade
            del_indices = np.random.choice(
                int(num_clients/2), size=delete_count, replace=False
            )
            search_indices = np.delete(np.arange(0, num_clients/2), del_indices)
        else:
            # Rodadas ímpares: remover clientes da segunda metade
            del_indices = np.random.choice(
                np.arange(num_clients/2, num_clients), size=delete_count, replace=False
            )
            search_indices = np.delete(np.arange(num_clients/2, num_clients), del_indices)
        
        # Normalizar ratios para clientes disponíveis
        available_ratios = client_ratios[search_indices.astype(int)]
        available_ratios = available_ratios / available_ratios.sum()
        
        # Selecionar clientes
        selected_indices = np.random.choice(
            search_indices.astype(int), 
            p=available_ratios, 
            size=self.clients_per_round, 
            replace=True
        )
        
        return selected_indices

    def _power_of_choice_intermittent(
        self, num_clients: int, client_ratios: np.ndarray, losses: np.ndarray
    ) -> np.ndarray:
        """Implementa power-of-choice com disponibilidade intermitente."""
        # Simular disponibilidade intermitente
        delete_count = int(self.delete_ratio * num_clients / 2)
        
        if (self.current_round % 2) == 0:
            del_indices = np.random.choice(
                int(num_clients/2), size=delete_count, replace=False
            )
            search_indices = np.delete(np.arange(0, num_clients/2), del_indices)
        else:
            del_indices = np.random.choice(
                np.arange(num_clients/2, num_clients), size=delete_count, replace=False
            )
            search_indices = np.delete(np.arange(num_clients/2, num_clients), del_indices)
        
        # Normalizar ratios para clientes disponíveis
        available_ratios = client_ratios[search_indices.astype(int)]
        available_ratios = available_ratios / available_ratios.sum()
        
        # Aplicar power-of-choice nos clientes disponíveis
        d = min(self.power_of_choice, len(search_indices))
        random_indices = np.random.choice(
            search_indices.astype(int), p=available_ratios, size=d, replace=False
        )
        
        # Ordenar por loss
        random_losses = [(losses[i], i) for i in random_indices]
        random_losses.sort(key=lambda x: x[0], reverse=True)
        
        # Selecionar top m clientes
        m = min(self.clients_per_round, len(random_losses))
        selected_indices = [idx for _, idx in random_losses[:m]]
        
        return np.array(selected_indices)

    def _afl_selection(self, num_clients: int, client_loss_proxy: np.ndarray) -> np.ndarray:
        """Implementa a estratégia Active Federated Learning."""
        soft_temp = 0.01
        proxy_copy = client_loss_proxy.copy()
        
        # Remover clientes com menor loss
        sorted_indices = np.argsort(proxy_copy)
        delete_count = int(self.delete_ratio * num_clients)
        
        for idx in sorted_indices[:delete_count]:
            proxy_copy[idx] = -np.inf
        
        # Calcular probabilidades
        exp_values = np.exp(soft_temp * proxy_copy)
        loss_prob = exp_values / exp_values.sum()
        
        # Seleção baseada em loss
        loss_based_count = int(np.floor((1 - self.rnd_ratio) * self.clients_per_round))
        idx1 = np.random.choice(
            num_clients, p=loss_prob, size=loss_based_count, replace=False
        )
        
        # Seleção aleatória do restante
        remaining_indices = np.delete(np.arange(num_clients), idx1)
        random_count = self.clients_per_round - loss_based_count
        idx2 = np.random.choice(remaining_indices, size=random_count, replace=False)
        
        return np.concatenate([idx1, idx2])

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Any]]:
        """Agrega os resultados do treinamento e atualiza métricas dos clientes."""
        
        # Atualizar métricas dos clientes
        for client, result in results:
            if result.metrics:
                # Assumindo que as métricas incluem 'loss'
                self.client_losses[client.cid] = result.metrics.get('loss', 0.0)
                self.client_loss_proxy[client.cid] = result.metrics.get('loss_proxy', 
                                                                        result.metrics.get('loss', 0.0))
        
        # Chamar agregação padrão
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """Agrega os resultados da avaliação."""
        return super().aggregate_evaluate(server_round, results, failures)
        
