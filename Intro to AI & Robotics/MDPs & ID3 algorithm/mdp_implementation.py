from copy import deepcopy
import numpy as np



def get_max_state(u, mdp, row, col):
    actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']
    max_val = float('-inf')
    max_action = None
    for action in actions:
        vals = []
        for i in range(4):
            p = mdp.transition_function[action][i]
            next = mdp.step((row, col), actions[i])
            if u[next[0]][next[1]] != 'WALL':
                vals.append(float(u[next[0]][next[1]]) * p)
        vals_sum = sum(vals)
        if vals_sum > max_val:
            max_val = vals_sum
            max_action = action
    ret = (max_val, max_action)
    return ret

def get_nonterminal_states(mdp, board):
    terminal_states = mdp.terminal_states
    ret = []
    for row in range(0, mdp.num_row):
        for col in range(0, mdp.num_col):
            if (row, col) not in terminal_states:
                ret.append((row, col, board[row][col]))
    return ret

def get_sum_u(u, mdp, row, col, action):
    actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']
    vals = []
    for i in range(4):
        p = mdp.transition_function[action][i]
        next = mdp.step((row, col), actions[i])
        if u[next[0]][next[1]] != 'WALL':
            vals.append(float(u[next[0]][next[1]]) * p)
    vals_sum = sum(vals)
    return vals_sum


actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']

def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    #
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #
    d = 1
    U = deepcopy(U_init)
    nonterminal_states = get_nonterminal_states(mdp, mdp.board)
    for state in mdp.terminal_states:
        i, j = state[0], state[1]
        U[i][j] = float(mdp.board[i][j])
    lim = epsilon * ((1 - mdp.gamma) / mdp.gamma)
    while d >= lim:
        d = 0
        prev_U = deepcopy(U)
        for i, j, val in nonterminal_states:
            if (val != 'WALL'):
                val = float(val)
                exp = get_max_state(prev_U, mdp, i, j)[0]
                U[i][j] = float(val + mdp.gamma * exp)
                prev_val = abs(U[i][j] - prev_U[i][j])
                d = max(d, prev_val)
    return U


def get_policy(mdp, U):
    #
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #
    rows = mdp.num_row
    cols = mdp.num_col
    policy = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if mdp.board[i][j] != 'WALL':
                v, a = get_max_state(U, mdp, i, j)

                policy[i][j] = a
    return policy


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']
    n = mdp.num_row * mdp.num_col
    P = np.zeros((n, n))
    R = np.zeros(n)

    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            state_idx = i*mdp.num_col + j
            action = policy[i][j]
            if (i, j) in mdp.terminal_states:
                R[state_idx] = mdp.board[i][j]
                continue

            if mdp.board[i][j] == 'WALL':
                R[state_idx] = 0
                continue
            for x in range(4):
                a = actions[x]
                p = mdp.transition_function[action][x]
                new_state = mdp.step((i, j), a)
                new_state_idx = new_state[0]*mdp.num_col + new_state[1]
                if mdp.board[new_state[0]][new_state[1]] != 'WALL' and mdp.board[i][j] != 'WALL':
                    P[state_idx, new_state_idx] += p
                    R[state_idx] = mdp.board[i][j]

    I = np.eye(n)
    U = np.linalg.solve((I - mdp.gamma * P), R)

    U = U.reshape(mdp.num_row, mdp.num_col)
    return U.tolist()


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #
    policy = policy_init
    states = get_nonterminal_states(mdp, mdp.board)
    changed = True
    while changed:
        u = policy_evaluation(mdp, policy_init)
        changed = False
        for state in mdp.terminal_states:
            row, col = state[0], state[1]
            u[row][col] = mdp.board[row][col]
        for state in states:
            row, col = state[0], state[1]
            if mdp.board[row][col] != 'WALL':
                new_val, action = get_max_state(u, mdp, row, col)
                policy_action = policy[row][col]
                prev_val = get_sum_u(u, mdp, row, col, policy_action)
                if new_val > prev_val:
                    policy[row][col] = action
                    changed = True
    return policy


"""For this functions, you can import what ever you want """

arrows = {
    'left': '\u2190',
    'right': '\u2192',
    'up': '\u2191',
    'down': '\u2193'
}

from mdp import MDP


def get_max_state_list(u, mdp, row, col, epsilon):
    actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']
    max_val = float('-inf')
    max_action = ""
    actions_values = []
    for action in actions:
        vals = []
        for i in range(4):
            p = mdp.transition_function[action][i]
            next = mdp.step((row, col), actions[i])
            if u[next[0]][next[1]] != 'WALL':
                vals.append(float(u[next[0]][next[1]]) * p)
        vals_sum = sum(vals)
        actions_values.append(vals_sum)
        if vals_sum >= max_val:
            max_val = vals_sum

    for x in range(4):
        if actions_values[x] >= max_val - epsilon:
            max_action += arrows[actions[x].lower()]
    num_of_options = len(max_action)
    ret = (num_of_options, max_action)
    return ret

def get_all_policies(mdp, U, epsilon=10 ** (-3)):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp, and the utility value U (which satisfies the Belman equation)
    # print / display all the policies that maintain this value
    # (a visualization must be performed to display all the policies)
    #
    # return: the number of different policies
    #
    rows = mdp.num_row
    cols = mdp.num_col
    policy = [[0 for _ in range(cols)] for _ in range(rows)]
    num_of_options = 1
    for row in range(rows):
        for col in range(cols):
            if mdp.board[row][col] != 'WALL' and (row, col) not in mdp.terminal_states:
                x, policy[row][col] = get_max_state_list(U, mdp, row, col, epsilon)
                num_of_options = num_of_options*x

    mdp.print_policy(policy)
    return num_of_options


def get_all_policies_ret(mdp, U, epsilon=10 ** (-3)):
    rows = len(U)
    cols = len(U[0])
    policy = [[0 for _ in range(cols)] for _ in range(rows)]
    num_of_options = 1
    for row in range(rows):
        for col in range(cols):
            if mdp.board[row][col] != 'WALL' and (row, col) not in mdp.terminal_states:
                x, policy[row][col] = get_max_state_list(U, mdp, row, col, epsilon)
                num_of_options = num_of_options*x
    return policy



def val_iteration(mdp, board, U_init, epsilon=10 ** (-3)):
    #
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #
    d = 1
    U = deepcopy(U_init)
    nonterminal_states = get_nonterminal_states(mdp, board)
    for state in mdp.terminal_states:
        i, j = state[0], state[1]
        U[i][j] = float(board[i][j])
    lim = epsilon * ((1 - mdp.gamma) / mdp.gamma)
    while d >= lim:
        d = 0
        prev_U = deepcopy(U)
        for i, j, val in nonterminal_states:
            if (val != 'WALL'):
                val = float(val)
                exp = get_max_state(prev_U, mdp, i, j)[0]
                U[i][j] = float(val + mdp.gamma * exp)
                prev_val = abs(U[i][j] - prev_U[i][j])
                d = max(d, prev_val)
    return U

def change_R(mdp, board, R):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if (i, j) not in mdp.terminal_states and board[i][j] != 'WALL':
                board[i][j] = R
    return board



def get_policy_for_different_rewards(mdp, epsilon=10 ** (-3)):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp
    # print / displas the optimal policy as a function of r
    # (reward values for any non-finite state)
    #

    R_values = []
    policies = []

    U = [[0 for _ in range(mdp.num_col)] for _ in range(mdp.num_row)]
    R = -5

    board = deepcopy(mdp.board)
    board = change_R(mdp, board, R)

    U = val_iteration(mdp, board, U)
    policy = get_all_policies_ret(mdp, U)
    prev_policy = policy

    R += 0.01
    while R <= 5:
        board = deepcopy(mdp.board)
        board = change_R(mdp, board, R)
        mdp_new = MDP(board=board,
                      terminal_states=mdp.terminal_states,
                      transition_function=mdp.transition_function,
                      gamma=mdp.gamma)
        U = value_iteration(mdp_new, U)
        new_policy = get_all_policies_ret(mdp_new, U, epsilon)
        if new_policy != policy:
            R_values.append(R)
            policies.append(new_policy)
            policy = new_policy
        R += 0.01

    R_values = np.round(R_values, 3)

    for r in range(len(R_values)):
        print("R(s) < " + str(R_values[r]))
        mdp.print_policy(prev_policy)
        prev_policy = policies[r]
        print(str(R_values[r]) + " <= ", end="")


    print("R(s)")
    mdp.print_policy(prev_policy)

    return R_values.tolist()

