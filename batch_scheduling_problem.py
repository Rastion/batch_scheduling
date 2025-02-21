import os
import random
from qubots.base_problem import BaseProblem
from math import comb  # for capacity offset (if needed)

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
            # Data: task resource assignment, task type, task duration, number of successors, successor task numbers
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
            if not tasks:
                return PENALTY
            d = self.duration[tasks[0]]
            # Check that all tasks in the batch have the same type and duration.
            for t in tasks:
                if self.types[t] != self.types[tasks[0]] or self.duration[t] != d:
                    return PENALTY
            # Batch processing time must equal the common duration.
            if (end - start) != d:
                return PENALTY
            # Capacity: number of tasks in a batch cannot exceed the resource capacity.
            if len(tasks) > self.capacity[r]:
                return PENALTY
            if end > makespan:
                makespan = end

        # For each resource, check that batches do not overlap.
        for r, batch_list in batches_by_resource.items():
            sorted_batches = sorted(batch_list, key=lambda b: b["start"])
            for i in range(1, len(sorted_batches)):
                if sorted_batches[i-1]["end"] > sorted_batches[i]["start"]:
                    return PENALTY

        # Precedence constraints: for each task i -> j, finish time of i <= start time of j.
        task_schedule = {}
        for batch in batches:
            for t in batch["tasks"]:
                task_schedule[t] = (batch["start"], batch["end"])
        for i in range(self.nb_tasks):
            for j in self.successors[i]:
                if i not in task_schedule or j not in task_schedule:
                    return PENALTY
                if task_schedule[i][1] > task_schedule[j][0]:
                    return PENALTY

        return makespan

    def random_solution(self):
        """
        Generate a random candidate solution.
        For simplicity, assign each task to its own batch with a random start time.
        """
        batches = []
        for t in range(self.nb_tasks):
            r = self.resources[t]
            d = self.duration[t]
            start = random.randint(0, max(0, self.time_horizon - d))
            batches.append({
                "resource": r,
                "tasks": [t],
                "start": start,
                "end": start + d
            })
        return {"batch_schedule": batches}

    def get_qubo(self, lambda_unique=10, lambda_prec=10, lambda_overlap=10, 
                 lambda_capacity=10, lambda_batch=10, objective_weight=1):
        """
        Construct a complete QUBO formulation of the scheduling problem.
        
        This method defines binary variables x_{t,s} for each task t and each allowed start time s.
        The QUBO includes the following penalty terms:
          1. Unique assignment: each task must have exactly one start time.
          2. Precedence: for every precedence constraint i->j, if task i starts at s and task j at s'
             with s + duration[i] > s', a penalty is added.
          3. Non-overlap: for tasks on the same resource (but not in the same batch) the later task's
             start time must be at least the earlier task's finish time.
          4. Capacity: for each resource r and each time slot s, if more than capacity[r] tasks are scheduled,
             a penalty is added (approximated by penalizing every pair of tasks assigned at the same time).
          5. Batch consistency: if tasks on the same resource are scheduled at the same time but have different
             type or duration, a penalty is added.
          
        Additionally, a linear objective term is added to minimize the surrogate completion time
        (s + duration[t]) for each task.
        
        The QUBO is returned as a dictionary mapping variable-index pairs (i,j) (with i<=j)
        to coefficients.
        
        This method also sets:
            - self.var_to_index: mapping from (t, s) -> unique variable index.
            - self.index_to_var: mapping from variable index -> (t, s).
        """
        self.var_to_index = {}  # (t, s) -> index
        self.index_to_var = {}  # index -> (t, s)
        index = 0
        Q = {}  # QUBO dictionary: keys are tuples (i, j) with i <= j

        def add_to_qubo(i, j, value):
            key = (min(i, j), max(i, j))
            Q[key] = Q.get(key, 0) + value

        # Build variable mapping: for each task t and allowed start time s.
        for t in range(self.nb_tasks):
            for s in range(0, self.time_horizon - self.duration[t] + 1):
                self.var_to_index[(t, s)] = index
                self.index_to_var[index] = (t, s)
                index += 1

        # 1. Unique assignment: (sum_s x_{t,s} - 1)^2
        for t in range(self.nb_tasks):
            valid_starts = range(0, self.time_horizon - self.duration[t] + 1)
            for i, s in enumerate(valid_starts):
                idx_i = self.var_to_index[(t, s)]
                add_to_qubo(idx_i, idx_i, -lambda_unique)
                for s2 in valid_starts[i+1:]:
                    idx_j = self.var_to_index[(t, s2)]
                    add_to_qubo(idx_i, idx_j, 2 * lambda_unique)

        # 2. Precedence constraints: for each i->j, if s_i + duration[i] > s_j then add penalty.
        for i in range(self.nb_tasks):
            for j in self.successors[i]:
                if j < 0 or j >= self.nb_tasks:
                    continue
                valid_starts_i = range(0, self.time_horizon - self.duration[i] + 1)
                valid_starts_j = range(0, self.time_horizon - self.duration[j] + 1)
                for s_i in valid_starts_i:
                    for s_j in valid_starts_j:
                        if s_i + self.duration[i] > s_j:
                            idx_i = self.var_to_index[(i, s_i)]
                            idx_j = self.var_to_index[(j, s_j)]
                            add_to_qubo(idx_i, idx_j, lambda_prec)

        # 3. Non-overlap constraint: for tasks on the same resource, batches must not overlap.
        # For every pair of tasks i and j (i < j) on the same resource and for every valid start times,
        # if one starts earlier but its finish time is later than the other's start, add a penalty.
        for i in range(self.nb_tasks):
            for j in range(i+1, self.nb_tasks):
                if self.resources[i] != self.resources[j]:
                    continue
                valid_starts_i = range(0, self.time_horizon - self.duration[i] + 1)
                valid_starts_j = range(0, self.time_horizon - self.duration[j] + 1)
                for s in valid_starts_i:
                    for s_j in valid_starts_j:
                        # If task i is scheduled before task j but overlaps
                        if s < s_j and s_j < s + self.duration[i]:
                            idx_i = self.var_to_index[(i, s)]
                            idx_j = self.var_to_index[(j, s_j)]
                            add_to_qubo(idx_i, idx_j, lambda_overlap)
                        # Also if task j is scheduled before task i but overlaps
                        elif s_j < s and s < s_j + self.duration[j]:
                            idx_i = self.var_to_index[(i, s)]
                            idx_j = self.var_to_index[(j, s_j)]
                            add_to_qubo(idx_i, idx_j, lambda_overlap)

        # 4. Capacity constraint: at each resource r and time slot s, penalize if more than capacity tasks are scheduled.
        # Here we add a pairwise penalty for every pair of tasks scheduled at time s on resource r.
        # (A proper zero-penalty for n <= capacity would require auxiliary variables.)
        for r in range(self.nb_resources):
            # Consider all time slots s in [0, time_horizon].
            for s in range(self.time_horizon):
                tasks_at_rs = [t for t in range(self.nb_tasks)
                               if self.resources[t] == r and s <= self.time_horizon - self.duration[t]]
                # Add quadratic penalty for every pair
                for i_idx in range(len(tasks_at_rs)):
                    for j_idx in range(i_idx+1, len(tasks_at_rs)):
                        t = tasks_at_rs[i_idx]
                        t_prime = tasks_at_rs[j_idx]
                        idx_i = self.var_to_index[(t, s)]
                        idx_j = self.var_to_index[(t_prime, s)]
                        add_to_qubo(idx_i, idx_j, lambda_capacity)
                # (Optionally, one can subtract a constant offset of lambda_capacity * comb(capacity[r], 2)
                # so that if at most capacity[r] tasks are scheduled, the net penalty is zero.
                # Since constant offsets do not change the optimizer’s argmin, we omit it here.)

        # 5. Batch consistency: tasks scheduled at the same time on the same resource must have the same type and duration.
        for r in range(self.nb_resources):
            for s in range(self.time_horizon):
                tasks_at_rs = [t for t in range(self.nb_tasks)
                               if self.resources[t] == r and s <= self.time_horizon - self.duration[t]]
                for i_idx in range(len(tasks_at_rs)):
                    for j_idx in range(i_idx+1, len(tasks_at_rs)):
                        t = tasks_at_rs[i_idx]
                        t_prime = tasks_at_rs[j_idx]
                        if self.types[t] != self.types[t_prime] or self.duration[t] != self.duration[t_prime]:
                            idx_i = self.var_to_index[(t, s)]
                            idx_j = self.var_to_index[(t_prime, s)]
                            add_to_qubo(idx_i, idx_j, lambda_batch)

        # 6. Objective: minimize surrogate finishing times (s + duration[t]) for each task.
        for t in range(self.nb_tasks):
            valid_starts = range(0, self.time_horizon - self.duration[t] + 1)
            for s in valid_starts:
                idx = self.var_to_index[(t, s)]
                add_to_qubo(idx, idx, objective_weight * (s + self.duration[t]))

        return Q

    def decode_solution(self, bitstring):
        """
        Decode a bitstring solution (list/array of 0/1 corresponding to QUBO variables) into
        the original batch scheduling format.
        
        For each task t, exactly one variable x_{t,s} should be 1. The chosen start times are then used
        to form batches. Tasks on the same resource with the same start time, type, and duration are grouped
        into a single batch.
        
        Returns:
            solution: A dictionary {"batch_schedule": [batch, ...]} where each batch is a dictionary with:
                        "resource", "tasks", "start", and "end".
            cost: The cost (makespan) computed by evaluate_solution.
        """
        # Step 1: Determine the assigned start time for each task.
        task_start = {}
        for t in range(self.nb_tasks):
            valid_starts = range(0, self.time_horizon - self.duration[t] + 1)
            # Find all start times with a 1 in the bitstring.
            chosen = [s for s in valid_starts if bitstring[self.var_to_index[(t, s)]] == 1]
            if len(chosen) != 1:
                # Either no start time or multiple start times were chosen for task t.
                #return None, float('inf')
                continue
            task_start[t] = chosen[0]
        
        # Step 2: Group tasks into batches.
        # We group tasks on the same resource that share the same (start time, type, duration).
        batches = []
        for r in range(self.nb_resources):
            # Get tasks assigned to resource r.
            tasks_on_r = [t for t in range(self.nb_tasks) if self.resources[t] == r]
            # Group tasks by (start, type, duration)
            groups = {}
            for t in tasks_on_r:
                key = (task_start[t], self.types[t], self.duration[t])
                groups.setdefault(key, []).append(t)
            # Create a batch for each group.
            for (s, ttype, d), tasks in groups.items():
                if len(tasks) > self.capacity[r]:
                    # Capacity violation: too many tasks grouped together.
                    #return None, float('inf')
                    continue
                batches.append({
                    "resource": r,
                    "tasks": tasks,
                    "start": s,
                    "end": s + d
                })
        
        solution = {"batch_schedule": batches}
        cost = self.evaluate_solution(solution)
        return solution, cost
