import os
import random
from qubots.base_problem import BaseProblem

class BatchSchedulingProblem(BaseProblem):
    """
    Batch Scheduling Problem

    In this problem, a set of tasks must be processed in batches on available resources.
    Each task is characterized by:
      - A fixed resource assignment.
      - A task type.
      - A processing duration.
      - Precedence (successor) constraints.
      
    A candidate solution is represented as a dictionary with one key:
      - "batch_schedule": a list of batches.
    
    Each batch is a dictionary with:
      - "resource": an integer (the resource on which the batch is processed),
      - "tasks": a list of task indices assigned to that batch,
      - "start": the start time (integer) of the batch,
      - "end": the end time (integer) of the batch.

    For a batch to be feasible, all tasks in the batch must have the same type and duration,
    and the number of tasks cannot exceed the capacity of the resource.
    
    Moreover, batches scheduled on the same resource must not overlap, and precedence constraints 
    between tasks must be satisfied.
    
    The objective is to minimize the makespan – the maximum finishing time among all batches.
    """

    def __init__(self, instance_file=None, 
                 nb_tasks=None, nb_resources=None, capacity=None, 
                 types=None, resources=None, duration=None, 
                 nb_successors=None, successors=None, 
                 nb_tasks_per_resource=None, time_horizon=None):
        if instance_file is not None:
            self._load_instance_from_file(instance_file)
        else:
            # Data provided directly
            if (nb_tasks is None or nb_resources is None or capacity is None or
                types is None or resources is None or duration is None or 
                nb_successors is None or successors is None or 
                nb_tasks_per_resource is None or time_horizon is None):
                raise ValueError("Either 'instance_file' or all instance parameters must be provided.")
            self.nb_tasks = nb_tasks
            self.nb_resources = nb_resources
            self.capacity = capacity
            self.types = types
            self.resources = resources
            self.duration = duration
            self.nb_successors = nb_successors
            self.successors = successors
            self.nb_tasks_per_resource = nb_tasks_per_resource
            self.time_horizon = time_horizon

    def _load_instance_from_file(self, filename):
        # Resolve relative paths with respect to this module's directory
        if not os.path.isabs(filename):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(base_dir, filename)
        with open(filename, 'r') as f:
            lines = f.readlines()
        # First line: number of tasks, number of resources
        first_line = lines[0].split()
        self.nb_tasks = int(first_line[0])
        self.nb_resources = int(first_line[1])
        # Second line: capacity for each resource
        second_line = lines[1].split()
        self.capacity = [int(x) for x in second_line]
        
        # Initialize lists to store task data
        self.types = []
        self.resources = []
        self.duration = []
        self.nb_successors = []
        self.successors = [[] for _ in range(self.nb_tasks)]
        # For grouping tasks per resource (optional)
        self.nb_tasks_per_resource = [0 for _ in range(self.nb_resources)]
        
        # Read each task (each subsequent line)
        for i in range(self.nb_tasks):
            line = lines[i+2].split()
            # According to the data: 
            # task resource assignment, task type, task duration, number of successors, successor task numbers
            self.resources.append(int(line[0]))
            self.types.append(int(line[1]))
            self.duration.append(int(line[2]))
            self.nb_successors.append(int(line[3]))
            for s in line[4:]:
                self.successors[i].append(int(s))
            # Count tasks per resource
            self.nb_tasks_per_resource[self.resources[i]] += 1
        
        # Trivial time horizon: sum of all durations
        self.time_horizon = sum(self.duration)

    def evaluate_solution(self, solution) -> float:
        """
        Evaluate a candidate solution.

        Candidate solution representation (dictionary):
          {
            "batch_schedule": [
                { "resource": int, "tasks": [int, ...], "start": int, "end": int },
                ...
            ]
          }

        Returns the makespan (max end time over all batches) if all constraints are met,
        otherwise returns a large penalty value.
        """
        PENALTY = 1e9
        if not isinstance(solution, dict) or "batch_schedule" not in solution:
            return PENALTY
        batches = solution["batch_schedule"]

        # Check that every task is assigned exactly once.
        assigned = [False] * self.nb_tasks
        for batch in batches:
            for t in batch.get("tasks", []):
                if t < 0 or t >= self.nb_tasks:
                    return PENALTY
                if assigned[t]:
                    return PENALTY
                assigned[t] = True
        if not all(assigned):
            return PENALTY

        makespan = 0
        # Group batches by resource to check non-overlap
        batches_by_resource = {}
        for batch in batches:
            r = batch.get("resource")
            if r is None or r < 0 or r >= self.nb_resources:
                return PENALTY
            batches_by_resource.setdefault(r, []).append(batch)
            # Check that batch start and end are integers and that end > start
            if not isinstance(batch.get("start"), int) or not isinstance(batch.get("end"), int):
                return PENALTY
            if batch["end"] <= batch["start"]:
                return PENALTY

        # Evaluate each batch
        for batch in batches:
            r = batch["resource"]
            tasks = batch["tasks"]
            start = batch["start"]
            end = batch["end"]
            # The batch processing time must equal the common duration of the tasks.
            if not tasks:
                return PENALTY
            d = self.duration[tasks[0]]
            # Check that all tasks in the batch have the same type and duration.
            for t in tasks:
                if self.types[t] != self.types[tasks[0]] or self.duration[t] != d:
                    return PENALTY
            # Also, the batch's duration should equal d.
            if (end - start) != d:
                return PENALTY
            # Capacity: number of tasks in batch cannot exceed resource capacity.
            if len(tasks) > self.capacity[r]:
                return PENALTY
            # Update makespan
            if end > makespan:
                makespan = end

        # For each resource, check that batches do not overlap.
        for r, batch_list in batches_by_resource.items():
            # Sort batches by start time.
            sorted_batches = sorted(batch_list, key=lambda b: b["start"])
            for i in range(1, len(sorted_batches)):
                prev = sorted_batches[i-1]
                curr = sorted_batches[i]
                if prev["end"] > curr["start"]:
                    return PENALTY

        # Precedence constraints:
        # For each task i that must precede task j, the batch containing i must finish
        # no later than the batch containing j starts.
        # Build a mapping from task to (batch start, batch end).
        task_schedule = {}
        for batch in batches:
            for t in batch["tasks"]:
                task_schedule[t] = (batch["start"], batch["end"])
        for i in range(self.nb_tasks):
            for j in self.successors[i]:
                # Adjust indices if instance uses 1-indexing for successors:
                # (Assuming here that successors are given in the instance as 0-indexed;
                # if they are 1-indexed, you would subtract 1.)
                if i not in task_schedule or j not in task_schedule:
                    return PENALTY
                # Enforce: end time of batch containing i <= start time of batch containing j.
                if task_schedule[i][1] > task_schedule[j][0]:
                    return PENALTY

        return makespan

    def random_solution(self):
        """
        Generate a random candidate solution.
        
        For simplicity, we assign each task to its own batch.
        The start time for each batch is chosen uniformly at random between 0 and (time_horizon - duration),
        and end time is start time plus the task's duration.
        """
        batches = []
        for t in range(self.nb_tasks):
            r = self.resources[t]  # resource is fixed per task from the instance
            d = self.duration[t]
            # Random start time in [0, time_horizon - d]
            start = random.randint(0, max(0, self.time_horizon - d))
            batch = {
                "resource": r,
                "tasks": [t],
                "start": start,
                "end": start + d
            }
            batches.append(batch)
        return {"batch_schedule": batches}
