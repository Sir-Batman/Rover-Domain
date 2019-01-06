from simulators.rover_domain_simulator import RoverDomain
from teams.rover_team import RoverTeam
from policies.policy import RandomPolicy
from rewards.g import GlobalReward
import yaml
import sys


def main():
    """ main

        This function contains the initial trial to run the rover domain.
    """
    # Read and store parameters from configuration file.
    if len(sys.argv) is 1:
        config_f = "config.yml"
    else:
        config_f = sys.argv[1]
    with open(config_f, 'r') as f:
        config_file = yaml.load(f)

    # Initialize the rover domain.
    domain = RoverDomain(config_file["Seed"], config_file["Initial POI Locations"],
                         config_file["Initial Agent Positions"], config_file["Number of Agents"],
                         config_file["Number of POIs"], config_file["World Width"], config_file["World Length"])

    # Intialize the rover team.
    # Rover team parameters
    agent_policies = dict()
    agent_policies_actions = dict()

    # For every agent, assign it a policy that maps input numpy vector x to dx,dy, currently just Random policy
    for i in range(config_file["Number of Agents"]):
        agent_policies["agent_" + str(i)] = RandomPolicy(2)
        agent_policies_actions["agent_" + str(i)] = agent_policies["agent_" + str(i)].get_next()

    use_distance = True

    team = RoverTeam(agent_policies_actions, use_distance)

    # Initialize the reward function
    global_reward = GlobalReward(config_file["Coupling"], config_file["Observation Radius"],
                                 config_file["Minimum Distance"])

    # Get States from Rover Doman
    joint_state = domain.get_jointstate()

    for step in range(config_file["Steps"]):
        print("Step:", step)

        print(joint_state['agents'])

        # Get the actions from the team
        actions = team.get_jointaction(joint_state)

        # Pass actions to domain to update
        domain.apply_actions(actions)

        # Update the joint state
        joint_state = domain.get_jointstate()

        # Compute the Global Reward
        global_reward.accept_jointstate(joint_state)
        reward_G = global_reward.calculate_reward()


if __name__ == '__main__':
    main()
