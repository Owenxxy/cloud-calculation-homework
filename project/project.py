import random
import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum


# Configuration
class Config:
    NUM_NODES = 50
    SIMULATION_STEPS = 100
    AGENT_COUNTS = [100, 500, 1000, 2000, 5000]
    RANDOM_SEED = 42
    CRITICAL_PERCENTAGE = 0.2
    CPU_CAPACITY = 100
    MAX_CPU_REQ_CRITICAL = (5, 15)
    MAX_CPU_REQ_BATCH = (1, 5)
    BASE_LATENCY_RANGE = (10, 50)
    NETWORK_CONGESTION_FACTOR = 200.0
    CONGESTION_PENALTY_MAX = 5.0
    INTENT_AWARE_PENALTY_REDUCTION = 0.8
    FAILURE_PENALTY_MULTIPLIER = 5


class SchedulerType(Enum):
    TRADITIONAL = "Traditional"
    INTENT_AWARE = "Intent-Aware"


class TaskType(Enum):
    CRITICAL = "CRITICAL"
    BATCH = "BATCH"


@dataclass
class Node:
    """Representation of a compute node in the system."""
    id: int
    cpu_capacity: int = Config.CPU_CAPACITY
    current_load: int = 0
    network_congestion: float = 0.0  # 0.0 to 1.0

    def reset(self) -> None:
        """Reset node state for a new simulation run."""
        self.current_load = 0
        self.network_congestion = 0.0

    def can_accept_task(self, cpu_req: int) -> bool:
        """Check if node has capacity for a task."""
        return self.current_load + cpu_req <= self.cpu_capacity

    def add_load(self, cpu_req: int) -> None:
        """Add task load to node and update congestion."""
        self.current_load += cpu_req
        self.network_congestion = min(
            1.0,
            self.current_load / Config.NETWORK_CONGESTION_FACTOR
        )

    def get_score(self) -> float:
        """Calculate node score for Intent-Aware scheduling."""
        return self.current_load + (self.network_congestion * 100)


@dataclass
class AgentTask:
    """Representation of an agent task."""
    id: int
    type: TaskType
    cpu_req: int
    base_latency: float

    @classmethod
    def create_random_task(cls, task_id: int) -> 'AgentTask':
        """Create a random task with appropriate distribution."""
        if random.random() < Config.CRITICAL_PERCENTAGE:
            task_type = TaskType.CRITICAL
            cpu_req = random.randint(*Config.MAX_CPU_REQ_CRITICAL)
        else:
            task_type = TaskType.BATCH
            cpu_req = random.randint(*Config.MAX_CPU_REQ_BATCH)

        base_latency = random.uniform(*Config.BASE_LATENCY_RANGE)
        return cls(task_id, task_type, cpu_req, base_latency)


class Scheduler:
    """Base class for different scheduling strategies."""

    def __init__(self, scheduler_type: SchedulerType):
        self.scheduler_type = scheduler_type

    def schedule_task(self, task: AgentTask, nodes: List[Node]) -> Tuple[Node, bool]:
        """Schedule a task to a node. Returns (selected_node, success)."""
        raise NotImplementedError

    def calculate_latency(self, task: AgentTask, node: Node, success: bool) -> float:
        """Calculate latency for a task based on scheduling result."""
        if not success:
            return task.base_latency * Config.FAILURE_PENALTY_MULTIPLIER

        congestion_penalty = 1.0 + (node.network_congestion * Config.CONGESTION_PENALTY_MAX)

        # Intent-Aware scheduler reduces penalty for critical tasks
        if (task.type == TaskType.CRITICAL and
                self.scheduler_type == SchedulerType.INTENT_AWARE):
            congestion_penalty *= Config.INTENT_AWARE_PENALTY_REDUCTION

        return task.base_latency * congestion_penalty


class TraditionalScheduler(Scheduler):
    """Traditional Round-Robin/Random scheduler."""

    def __init__(self):
        super().__init__(SchedulerType.TRADITIONAL)

    def schedule_task(self, task: AgentTask, nodes: List[Node]) -> Tuple[Node, bool]:
        """Randomly select a node."""
        selected_node = random.choice(nodes)
        success = selected_node.can_accept_task(task.cpu_req)
        if success:
            selected_node.add_load(task.cpu_req)
        return selected_node, success


class IntentAwareScheduler(Scheduler):
    """Intent-Aware scheduler that considers task types and node conditions."""

    def __init__(self):
        super().__init__(SchedulerType.INTENT_AWARE)

    def schedule_task(self, task: AgentTask, nodes: List[Node]) -> Tuple[Node, bool]:
        """Select node with lowest load and congestion score."""
        best_node = None
        best_score = float('inf')

        for node in nodes:
            if node.can_accept_task(task.cpu_req):
                score = node.get_score()
                if score < best_score:
                    best_score = score
                    best_node = node

        if best_node:
            best_node.add_load(task.cpu_req)
            return best_node, True
        else:
            # If no node can accept, return first node (will fail)
            return nodes[0], False


def generate_tasks(agent_count: int) -> List[AgentTask]:
    """Generate a list of tasks for simulation."""
    tasks = []
    for i in range(agent_count):
        tasks.append(AgentTask.create_random_task(i))
    return tasks


def prioritize_tasks(tasks: List[AgentTask]) -> List[AgentTask]:
    """Prioritize critical tasks over batch tasks."""
    return sorted(tasks, key=lambda x: 0 if x.type == TaskType.CRITICAL else 1)


def run_simulation(scheduler_type: SchedulerType, agent_count: int, nodes: List[Node]) -> Tuple[float, int]:
    """Run a single simulation and return average latency and failure count."""
    # Reset all nodes
    for node in nodes:
        node.reset()

    # Generate tasks
    tasks = generate_tasks(agent_count)

    # Create appropriate scheduler
    if scheduler_type == SchedulerType.TRADITIONAL:
        scheduler = TraditionalScheduler()
    else:
        scheduler = IntentAwareScheduler()
        # Intent-Aware prioritizes critical tasks
        tasks = prioritize_tasks(tasks)

    total_latency = 0.0
    failures = 0

    # Process all tasks
    for task in tasks:
        selected_node, success = scheduler.schedule_task(task, nodes)
        latency = scheduler.calculate_latency(task, selected_node, success)
        total_latency += latency

        if not success:
            failures += 1

    avg_latency = total_latency / agent_count if agent_count > 0 else 0.0
    return avg_latency, failures


def plot_results(agent_counts: List[int], results_traditional: List[float],
                 results_intent: List[float], failures_trad: List[int],
                 failures_intent: List[int]) -> None:
    """Create visualization of simulation results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Latency comparison
    ax1.plot(agent_counts, results_traditional, marker='o', linestyle='--',
             label='Traditional (Round-Robin)', color='red', linewidth=2)
    ax1.plot(agent_counts, results_intent, marker='s', linewidth=2,
             label='Intent-Aware (Proposed)', color='blue')

    ax1.set_title('Average Agent Task Latency vs. Scale', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Concurrent Agents', fontsize=12)
    ax1.set_ylabel('Average Latency (ms)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.tick_params(labelsize=10)

    # Add annotations for significant improvements
    for i, (trad, intent) in enumerate(zip(results_traditional, results_intent)):
        improvement = ((trad - intent) / trad) * 100
        if improvement > 10:  # Only show significant improvements
            ax1.annotate(f'{improvement:.1f}%',
                         xy=(agent_counts[i], intent),
                         xytext=(0, -20),
                         textcoords='offset points',
                         ha='center',
                         fontsize=9,
                         color='green',
                         fontweight='bold')

    # Plot 2: Failure comparison
    x = np.arange(len(agent_counts))
    width = 0.35

    ax2.bar(x - width / 2, failures_trad, width, label='Traditional',
            color='red', alpha=0.7)
    ax2.bar(x + width / 2, failures_intent, width, label='Intent-Aware',
            color='blue', alpha=0.7)

    ax2.set_title('Task Failures vs. Scale', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Concurrent Agents', fontsize=12)
    ax2.set_ylabel('Number of Failures', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(count) for count in agent_counts])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.tick_params(labelsize=10)

    plt.tight_layout()

    # Save plot
    output_path = 'latency_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {os.path.abspath(output_path)}")

    # Also display the plot
    plt.show()


def print_results_table(agent_counts: List[int], results_trad: List[float],
                        results_intent: List[float], failures_trad: List[int],
                        failures_intent: List[int]) -> None:
    """Print formatted results table."""
    print("\n" + "=" * 80)
    print(f"{'Simulation Results':^80}")
    print("=" * 80)
    print(f"{'Agents':<12} {'Traditional':<20} {'Intent-Aware':<20} {'Improvement':<15}")
    print(f"{'':<12} {'Latency(ms)':<10} {'Failures':<10} {'Latency(ms)':<10} {'Failures':<10} {'%':<15}")
    print("-" * 80)

    for i, count in enumerate(agent_counts):
        trad_lat = results_trad[i]
        intent_lat = results_intent[i]
        trad_fail = failures_trad[i]
        intent_fail = failures_intent[i]

        if trad_lat > 0:
            improvement = ((trad_lat - intent_lat) / trad_lat) * 100
        else:
            improvement = 0.0

        print(
            f"{count:<12} {trad_lat:<10.2f} {trad_fail:<10} {intent_lat:<10.2f} {intent_fail:<10} {improvement:<15.1f}")


def main():
    """Main simulation function."""
    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)

    # Initialize nodes
    nodes = [Node(i) for i in range(Config.NUM_NODES)]

    # Results storage
    results_traditional = []
    results_intent = []
    failures_traditional = []
    failures_intent = []

    print(f"Running simulation with {Config.NUM_NODES} nodes...")
    print(f"Testing agent counts: {Config.AGENT_COUNTS}")
    print("-" * 60)

    # Run simulations for each agent count
    for count in Config.AGENT_COUNTS:
        print(f"\nSimulating {count} agents...")

        lat_trad, fail_trad = run_simulation(
            SchedulerType.TRADITIONAL, count, nodes
        )
        lat_intent, fail_intent = run_simulation(
            SchedulerType.INTENT_AWARE, count, nodes
        )

        results_traditional.append(lat_trad)
        results_intent.append(lat_intent)
        failures_traditional.append(fail_trad)
        failures_intent.append(fail_intent)

        print(f"  Traditional: {lat_trad:.2f}ms, {fail_trad} failures")
        print(f"  Intent-Aware: {lat_intent:.2f}ms, {fail_intent} failures")

    # Display results
    print_results_table(
        Config.AGENT_COUNTS,
        results_traditional,
        results_intent,
        failures_traditional,
        failures_intent
    )

    # Plot results
    plot_results(
        Config.AGENT_COUNTS,
        results_traditional,
        results_intent,
        failures_traditional,
        failures_intent
    )

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(
        f"Average Latency Improvement: {np.mean([((t - i) / t) * 100 for t, i in zip(results_traditional, results_intent) if t > 0]):.1f}%")
    print(f"Total Failures Reduced: {sum(failures_traditional) - sum(failures_intent)}")
    print(f"Max Latency Reduction: {max(results_traditional) - max(results_intent):.2f}ms")


if __name__ == "__main__":
    main()