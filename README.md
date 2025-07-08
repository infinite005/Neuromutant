# Neuromutant

A reinforcement learning project that evolves its own source code by applying mutations using Abstract Syntax Tree (AST) transformations. The agent uses Q-learning to learn which code mutations improve the correctness of a target function.

---

## Features

- **AST-based code mutation:** Mutates Python source code safely using AST.
- **Reinforcement learning:** Uses Q-learning with a replay buffer to select and evaluate mutations.
- **Self-modifying:** The script overwrites its own source code when successful mutations are found.
- **Simple testing framework:** Tests the correctness of a sample function (`add`) after each mutation to provide reward signals.

---

## How It Works

1. **Initial State:** The Q-table starts with a single state and zero values for all actions.
2. **Mutation Actions:** The agent can perform actions like incrementing integer constants or renaming variables.
3. **Testing Mutations:** After mutation, the code is executed and tested against predefined test cases.
4. **Reward Signal:** Rewards guide the agent to learn which mutations lead to correct and improved code.
5. **Q-learning Updates:** The Q-table is updated using experiences stored in a replay buffer.
6. **Self-Update:** When a mutation yields a positive reward, the script rewrites itself with the new code and restarts.

---

## Requirements

- Python 3.7+
- `numpy`
- `astunparse`

Install dependencies via pip:

```bash
pip install numpy astunparse

## Usage
Simply run the script:
python main.py
The script will continually mutate and improve itself, printing out mutation summaries and rewards.

## Project Structure
-main.py — The main self-modifying Q-learning agent script
q_table.npy — Saved Q-table data (generated after running)

## Note
The state space is currently simple and grows by one each time a mutation occurs.
Actions and testing logic can be extended to support more complex mutations and evaluation.
Be careful when running self-modifying code scripts; always keep backups!

## License
MIT License © Mohammad Samadi

## Contact
For questions or contributions, open an issue or contact me at: mohammad.rz.samadi@gmai.com




