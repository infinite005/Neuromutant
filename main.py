import ast
import astunparse
import os
import sys
import random
import numpy as np
import io
import contextlib

# ================= Replay Buffer ===================
class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = []
        self.capacity = capacity

    def add(self, state, action, reward, next_state):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self):
        return len(self.buffer)

# ================ RL Settings =======================
actions = ["increment_constants", "rename_variable"]

if os.path.exists("q_table.npy"):
    q_table = np.load("q_table.npy")
else:
    q_table = np.zeros((1, len(actions)))  # Start with one state

learning_rate = 0.1
discount = 0.95
exploration_rate = 0.2
batch_size = 4
state = 0  # current state
next_state = 0  # next state placeholder

replay_buffer = ReplayBuffer(500)

# Function to ensure the Q-table has enough rows for the given state
def ensure_state_exists(state):
    global q_table
    if state >= q_table.shape[0]:
        new_rows = state - q_table.shape[0] + 1
        q_table = np.vstack([q_table, np.zeros((new_rows, len(actions)))])

# ================ Code Mutation ==========================
def mutate_code(source_code, action):
    tree = ast.parse(source_code)
    change_log = []

    class Mutator(ast.NodeTransformer):
        def visit_Constant(self, node):
            if action == "increment_constants" and isinstance(node.value, int):
                change_log.append(f"Replaced constant {node.value} → {node.value + 1}")
                return ast.copy_location(ast.Constant(value=node.value + 1), node)
            return node

        def visit_Name(self, node):
            if action == "rename_variable" and node.id == "x":
                change_log.append("Renamed variable 'x' → 'y'")
                return ast.copy_location(ast.Name(id="y", ctx=node.ctx), node)
            return node

        def visit_assign(self, node):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                new_val = node.value.value + 1
                node.value = ast.Constant(value=new_val)
            return node

    new_tree = Mutator().visit(tree)
    ast.fix_missing_locations(new_tree)
    mutated_code = astunparse.unparse(new_tree)
    return mutated_code, change_log

# ================ Code Testing ============================
def test_code(code):
    try:
        local_env = {}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(code, local_env)

        if "add" in local_env:
            tests = [
                ((1, 2), 3),
                ((5, 7), 12),
                ((-1, 1), 0)
            ]
            for args, expected in tests:
                result = local_env["add"](*args)
                if result != expected:
                    return 0
            return 10
        return -1
    except Exception:
        return -10

# =============== File Reading / Writing ========================
def read_self():
    with open(__file__, "r", encoding="utf-8") as f:
        return f.read()

def write_self(new_code):
    with open(__file__, "w", encoding="utf-8") as f:
        f.write(new_code)

# =============== Action Selection ========================
def choose_action():
    global state
    ensure_state_exists(state)  # Ensure current state exists in Q-table
    if random.uniform(0, 1) < exploration_rate:
        return random.randint(0, len(actions) - 1)
    return int(np.argmax(q_table[state]))

# =============== Example Function to be Tested ========================
def add(a, b):
    return a + b

# =============== Main Learning Loop ======================
if __name__ == "__main__":
    
    source = read_self()

    ensure_state_exists(state)    # Make sure current state exists

    action_idx = choose_action()
    action = actions[action_idx]

    mutated_code, change_log = mutate_code(source, action)
    reward = test_code(mutated_code)

    # Update next_state as current state + 1 (you can modify this logic as needed)
    next_state = state + 1

    ensure_state_exists(next_state)  # Ensure next state exists in Q-table

    replay_buffer.add(state, action_idx, reward, next_state)

    batch = replay_buffer.sample(batch_size)
    for s, a, r, ns in batch:
        ensure_state_exists(s)
        ensure_state_exists(ns)
        old_q = q_table[s, a]
        next_max = np.max(q_table[ns])
        new_q = old_q + learning_rate * (r + discount * next_max - old_q)
        q_table[s, a] = new_q

    np.save("q_table.npy", q_table)

    # Update current state to next_state for next iteration
    state = next_state

    if reward > 0:
        print(f"✅ The mutation '{action}' was successful! Reward: {reward}")
        if change_log:
            print("📢 Change Summary:")
            for log in change_log:
                print(f"- {log}")
        write_self(mutated_code)
        os.execv(sys.executable, ["python"] + sys.argv)
    else:
        print(f"❌ The mutation '{action}' failed! Reward: {reward}")
        print("Q-table:", q_table)
        if change_log:
            print("🔎 Attempted Changes:")
            for log in change_log:
                print(f"- {log}")
