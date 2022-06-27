import argparse
import torch
from sac import SAC
from trainer import Trainer
from utils import set_seed, make_env, make_delayed_env


def get_parameters():
    parser = argparse.ArgumentParser()

    # ---------------Delay 관련 노브---------------
    parser.add_argument('--env-name', default='Walker2d-v3')
    parser.add_argument('--delay_step', default=1)
    parser.add_argument('--rollout', default=20000)  # 몇 step 마다 커널을 업데이트 할 것인가?

    # --------------------------------------------
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--state-dim', default=None)
    parser.add_argument('--action-dim', default=None)
    parser.add_argument('--action_bound', default=[None, None])

    parser.add_argument('--random-seed', default=-1, type=int)
    parser.add_argument('--eval_flag', default=True, type=bool)
    parser.add_argument('--eval-freq', default=5000, type=int)
    parser.add_argument('--eval-episode', default=5, type=int)

    parser.add_argument('--automating-temperature', default=True, type=bool)
    parser.add_argument('--temperature', default=0.2, type=float)
    parser.add_argument('--start-step', default=20000, type=int)
    parser.add_argument('--max-step', default=500000, type=int)
    parser.add_argument('--update_after', default=1000, type=int)
    parser.add_argument('--hidden-dims', default=(256, 256))
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--buffer-size', default=1000000, type=int)
    parser.add_argument('--update-every', default=50, type=int)
    parser.add_argument('--log_std_bound', default=[-20, 2])
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--actor-lr', default=3e-4, type=float)
    parser.add_argument('--critic-lr', default=3e-4, type=float)
    parser.add_argument('--temperature-lr', default=3e-4, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--log', default=True, type=bool)
    parser.add_argument('--show-loss', default=False, type=bool)

    param = parser.parse_args()

    return param


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    delays = [1, 2, 3]
    seeds = [1, 2, 3, 4, 5]
    for delay in delays:
        for seed in seeds:
            args.delay_step = delay
            args.random_seed = seed
            random_seed = set_seed(args.random_seed)

            env, eval_env = make_delayed_env(args, random_seed, delay_step=args.delay_step)

            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            action_bound = [env.action_space.low[0], env.action_space.high[0]]
            args.state_dim = state_dim
            args.action_dim = action_dim
            args.action_bound = action_bound

            print("Device:", device, "\nRandom Seed:", random_seed, "\nEnvironment:", args.env_name,
                  '\nAdjust Temperature Automatically:', args.automating_temperature, '\nActor_lr:', args.actor_lr,
                  '\nCritic_lr:', args.critic_lr, '\nTemperature_lr', args.temperature_lr, '\nBatch_size:', args.batch_size,
                  '\nDelay_step:', args.delay_step, '\n')

            agent = SAC(args, state_dim, action_dim, action_bound, env.action_space, device)

            trainer = Trainer(env, eval_env, agent, args)
            trainer.run()


if __name__ == '__main__':
    args = get_parameters()
    main(args)
