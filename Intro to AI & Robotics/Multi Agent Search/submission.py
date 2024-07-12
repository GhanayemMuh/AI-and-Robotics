import time
import math
from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random


# TODO: section a : 3
'''''
def choose_package(dist1, dist2, credit_mod):
    first_package_better = (dist1[0] <= dist1[1]) and \
                           (dist1[0] <= dist2[0] or \
                            dist2[1] <= dist2[0])

    second_package_better = (dist1[1] < dist1[0] and \
                             dist1[1] <= dist2[1]) or \
                            (dist2[0] < dist1[0] and \
                             dist2[0] <= dist2[1])

    if first_package_better:
        return -(dist1[0] - credit_mod)
    elif second_package_better:
        return -(dist1[1] - credit_mod)
    else:
        return -(dist1[0] - credit_mod)


def smart_heuristic(env: WarehouseEnv, robot_id):
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)
    credit_mod = 14 * robot.credit - (robot.credit - other_robot.credit)

    if robot.package is not None:
        return -(manhattan_distance(robot.position, robot.package.destination) - credit_mod)

    distance_from_pckgs = [manhattan_distance(robot.position, package.position) +
                           manhattan_distance(package.position, package.destination)
                           for package in env.packages]

    other_robot_distance_from_pckgs = [manhattan_distance(other_robot.position, package.position) +
                                       manhattan_distance(package.position, package.destination)
                                       for package in env.packages]

    # Only 1 package left.
    if len(distance_from_pckgs) == 1:
        return -(distance_from_pckgs[0] - credit_mod)

    return choose_package(distance_from_pckgs, other_robot_distance_from_pckgs, credit_mod)



'''
def going_to(env: WarehouseEnv, robot):
   # robot = env.get_robot(robot_id)
    robot_pos = robot.position
    robot_battery = robot.battery
    package_pos1 = env.packages[0].position
    package_dest1 = env.packages[0].destination
    package_pos2 = env.packages[1].position
    package_dest2 = env.packages[1].destination
    dist1 = manhattan_distance(robot_pos, package_pos1)
    dist11 = manhattan_distance(robot_pos, package_dest1)
    dist2 = manhattan_distance(robot_pos, package_pos2)
    dist22 = manhattan_distance(robot_pos, package_dest2)

    if dist2 == dist1:
        return (-1, -1)
    if dist11 < min(dist22, dist2) or dist1 < min(dist22, dist2):
        return package_pos1
    if dist22 < min(dist11, dist1) or dist2 < min(dist11, dist1):
        return package_pos2
    if dist22 < min(dist11, dist2, dist1):
        return package_pos2
    if dist1 < dist2 and dist1 < robot_battery:
        return package_pos1
    elif dist2 < dist1 and dist2 < robot_battery:
        return package_pos2
    else:
        return (-1, -1)


def best_package(env: WarehouseEnv, robot, other_robot):
    g0= manhattan_distance(env.packages[0].position, env.packages[0].destination)
    g1 = manhattan_distance(env.packages[1].position, env.packages[1].destination)



    if g0 > g1:
       return (env.packages[0].position, env.packages[0].destination)
    elif g1 > g0:
        return (env.packages[1].position, env.packages[1].destination)
    else:
        if manhattan_distance(robot.position, env.packages[0].position) > manhattan_distance(robot.position, env.packages[1].position):
            return (env.packages[1].position, env.packages[1].destination)
        else:
            return (env.packages[0].position, env.packages[0].destination)

def min_dist_to_charge(env: WarehouseEnv, robot):
    c0 = env.charge_stations[0].position
    c1 = env.charge_stations[1].position
    dist0 = manhattan_distance(robot.position, c0)
    dist1 = manhattan_distance(robot.position, c1)
    return min(dist0, dist1)


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id+1)%2)
    if robot.package is not None:
        if robot.battery > manhattan_distance(robot.package.destination, robot.position):
            return manhattan_distance(robot.package.position, robot.package.destination) -\
                   2*manhattan_distance(robot.position, robot.package.destination) + 100*robot.credit + manhattan_distance(robot.position, other_robot.position)/2+500
        elif manhattan_distance(robot.position, robot.package.destination) + 2 < robot.credit:
            return robot.battery*100 + robot.credit*1000 - min_dist_to_charge(env, robot)
        else:
            return manhattan_distance(robot.package.position, robot.package.destination) - \
                  2*manhattan_distance(robot.position, robot.package.destination) + 1000 * robot.credit

    else:
        package = best_package(env, robot, other_robot)
        if manhattan_distance(robot.position, package[0]) + manhattan_distance(package[0], package[1]) + 2 < robot.battery:
            return 1000*robot.credit - 2*manhattan_distance(robot.position, package[0]) + manhattan_distance(robot.position, other_robot.position)/2
        elif manhattan_distance(robot.position, package[0]) + manhattan_distance(package[0], package[1]) + 2 < robot.credit:
            return robot.battery*1000 + robot.credit*1000 - min_dist_to_charge(env, robot)
        return 1000*robot.credit - 2*manhattan_distance(robot.position, package[0])



class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)



class AgentMinimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        best_operator = None
        depth = 0
        try:
            while depth < 7:
                best_operator = self.minimax_aux(env, agent_id, time_limit, start_time, depth)
                depth += 1
        except:
            return best_operator

        return best_operator

    def minimax_aux(self, env: WarehouseEnv, agent_id, time_limit, start_time, depth):
        operators, children = self.successors(env, agent_id)
        children_heuristics = [self.minimax(child, (agent_id+1)%2, False, time_limit, start_time, depth) for child in children]
        max_h = max(children_heuristics)
        possible_ops = [i for i, c in enumerate(children_heuristics) if c == max_h]
        operator = operators[random.choice(possible_ops)]
        return operator
        '''''
        for op, child in zip(operators, children):
            score = self.minimax(child, (agent_id+1)%2, False, time_limit, start_time, depth)
            if score > best_score:
                best_score = score
                best_op = op
        return best_op
        '''''

    def minimax(self, env: WarehouseEnv, agent_id, maximizing_player, time_limit, start_time, depth):
        time_spent = time.time() - start_time
        if time_spent >= time_limit - 0.1:
            raise Exception("too much time")

        if depth == 0 or env.done():
            if maximizing_player:
                return smart_heuristic(env, agent_id)
            else:
                return smart_heuristic(env, (agent_id+1)%2)

        operators, children = self.successors(env, agent_id)

        if maximizing_player:
            best_score = float('-inf')
            for op, child in zip(operators, children):
                score = self.minimax(child, (agent_id+1)%2, False, time_limit, start_time, depth - 1)
                if score > best_score:
                    best_score = score
            return best_score
        else:
            best_score = float('inf')
            for op, child in zip(operators, children):
                score= self.minimax(child, (agent_id+1)%2, True, time_limit, start_time, depth - 1 )
                if score < best_score:
                    best_score = score

            return best_score

class AgentAlphaBeta(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        best_operator = None
        depth = 0
        try:
            while depth < 7:
                best_operator = self.alphabeta_aux(env, agent_id, time_limit, start_time, depth)
                depth += 1
        except:
            return best_operator
        return best_operator

    def alphabeta_aux(self, env: WarehouseEnv, agent_id, time_limit, start_time, depth):
        operators, children = self.successors(env, agent_id)

        children_heuristics = [self.alphabeta(child, (agent_id+1)%2, False, time_limit, start_time, depth, -math.inf, math.inf) for child in children]
        max_h = max(children_heuristics)
        possible_ops = [i for i, c in enumerate(children_heuristics) if c == max_h]
        operator = operators[random.choice(possible_ops)]
        return operator
        '''''
        best_score = float('-inf')
        best_op = None
        for op, child in zip(operators, children):
            score = self.alphabeta(child, (agent_id+1)%2, False, time_limit, start_time, depth, -math.inf, math.inf)
            if score > best_score:
                best_score = score
                best_op = op
        return best_op
        '''''


    def alphabeta(self, env: WarehouseEnv, agent_id, maximizing_player, time_limit, start_time, depth, alpha, beta):
        time_spent = time.time() - start_time
        if time_spent >= time_limit - 0.1:
            raise Exception("too much time")

        if depth == 0 or env.done():
            if maximizing_player:
                return smart_heuristic(env, agent_id)
            else:
                return smart_heuristic(env, (agent_id+1)%2)

        operators, children = self.successors(env, agent_id)

        if maximizing_player:
            best_score = float('-inf')
            for op, child in zip(operators, children):
                score = self.alphabeta(child, (agent_id+1)%2, False, time_limit, start_time, depth - 1, alpha, beta)
                if score > best_score:
                    best_score = score
                alpha = max(best_score, alpha)
                if best_score >= beta:
                    return math.inf
            return best_score
        else:
            best_score = float('inf')
            for op, child in zip(operators, children):
                score= self.alphabeta(child, (agent_id+1)%2, True, time_limit, start_time, depth - 1, alpha, beta)
                if score < best_score:
                    best_score = score
                beta = min(best_score, beta)
                if best_score <= alpha:
                    return -math.inf
            return best_score



''''
class AgentAlphaBeta(Agent):
    def __init__(self):
        self.agent_id = None

    def successors(self, env: WarehouseEnv, robot_id: int):
        operators = env.get_legal_operators(robot_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(robot_id, op)
        return operators, children

    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

    def rb_alpha_beta(self, env, agent_id, time_limit, depth, alpha, beta):
        if time.time() > time_limit - 0.1 or env.done() or depth == 0:
            return self.heuristic(env, agent_id), None
        operators, children = self.successors(env, agent_id)
        action = operators[0]
        if self.agent_id == agent_id:
            curr_max = float('-inf')
            for child, op in zip(children, operators):
                if time.time() > time_limit - 0.1:
                    break
                hueristic_val, _ = self.rb_alpha_beta(child, agent_id, time_limit, depth - 1, alpha, beta)
                if hueristic_val > curr_max:
                    curr_max = hueristic_val
                    action = op
                alpha = max(alpha, curr_max)
                if beta <= alpha:
                    break
            return curr_max, action
        else:
            curr_min = float('inf')
            for child, op in zip(children, operators):
                if time.time() > time_limit - 0.1:
                    break
                hueristic_val, _ = self.rb_alpha_beta(child, agent_id, time_limit, depth - 1, alpha, beta)
                if hueristic_val < curr_min:
                    curr_min = hueristic_val
                    action = op
                beta = min(beta, curr_min)
                if beta <= alpha:
                    break
            return curr_min, action

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.agent_id = agent_id
        depth = 1
        best_score = float('-inf')
        best_action = None
        start_time = time.time()
        end_time = start_time + time_limit

        while time.time() < end_time - 0.1:
            score, action = self.rb_alpha_beta(env, agent_id, end_time, depth, float('-inf'), float('inf'))
            if score > best_score:
                best_score = score
                best_action = action
            depth += 1

        return best_action

'''
class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        best_operator = None

        depth = 0

        try:
            while True:
                best_operator = self.expectimax_aux(env, agent_id, time_limit, start_time, depth)
                depth += 1

        except:
            return best_operator

    def probability(self, operators):
        ret = 0
        for op in operators:
            if op == "move east" or op == "pick up":
                ret += 2
            else:
                ret += 1
        return ret

    def expectimax_aux(self, env: WarehouseEnv, agent_id, time_limit, start_time, depth):
        operators, children = self.successors(env, agent_id)
        best_score = float('-inf')
        best_op = None
        for op, child in zip(operators, children):
            score = self.expectimax(child, (agent_id + 1) % 2, False, time_limit, start_time, depth)
            if score > best_score:
                best_score = score
                best_op = op
        return best_op

    def expectimax(self, env: WarehouseEnv, agent_id, maximizing_player, time_limit, start_time, depth):
        time_spent = time.time() - start_time
        if time_spent >= time_limit - 0.1:
            raise Exception("too much time")
        if depth == 0 or env.done():
            if maximizing_player:
                return smart_heuristic(env, agent_id)
            else:
                return smart_heuristic(env, (agent_id+1)%2)

        operators, children = self.successors(env, agent_id)

        if maximizing_player:
            best_score = float('-inf')

            # Perform maximization
            for child in children:
                score = self.expectimax(child, (agent_id+1)%2, False, time_limit, start_time, depth - 1)
                best_score = max(best_score, score)

            return best_score
        else:
            p = self.probability(operators)
            exp = 0
            for op, child in zip(operators, children):
                score = self.expectimax(child, (agent_id+1)%2, True, time_limit, start_time, depth-1)
                if op == "move east" or op == "pick up":
                    exp += (2*score)/p
                else:
                    exp += score/p

            return exp



# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move east", "pick up", "move east", "move east", "move south", "move south", "move south",
                           "move south", "drop off", "move north", "move north", "move west", "move west", "pick up", "move west", "move north", "drop off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
