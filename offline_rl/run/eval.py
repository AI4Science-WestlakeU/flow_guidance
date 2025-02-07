import os
import tqdm
import tyro
import torch
from gflower.config.flow_matching import FlowMatchingEvaluationConfig
from gflower.datasets.sequence import SequenceDataset
from gflower.models_flow.flow_policy import FlowPolicy
from gflower.models_flow.transformer import TransformerFlow
from gflower.models_value.transformer import Transformer as ValueTransformer
from run.utils import deterministic, save_config, set_cuda_visible_device
import numpy as np

def run_env(env, policy, cfg: FlowMatchingEvaluationConfig):
    observation = env.reset()

    rollout = [observation.copy()] # for rendering
    total_reward = 0
    # Create tqdm progress bar with reward and score as postfix
    pbar = tqdm.tqdm(range(cfg.max_episode_length), desc='Episode')
    for t in pbar:
        state = env.state_vector().copy()
        conditions = {0: observation}

        # this policy is "replan-1" in CL_DiffPhyCon
        action, samples = policy(conditions, batch_size=cfg.batch_size)

        next_observation, reward, terminal, _ = env.step(action) # TODO: make compatible with gymnasium


        total_reward += reward
        score = env.get_normalized_score(total_reward)
        rollout.append(next_observation.copy())
        observation = next_observation

        # Update progress bar postfix with reward and score
        pbar.set_postfix({'reward': f'{reward:.3f}', 'score': f'{score:.3f}'})
    
    return score
    
def evaluate(cfg: FlowMatchingEvaluationConfig):
    # get policy

    # get dataset only for its normalizer
    dataset = SequenceDataset(
        env=cfg.env,
        horizon=cfg.horizon,
        normalizer=cfg.normalizer,
        preprocess_fns=cfg.preprocess_fns,
        max_path_length=cfg.max_path_length,
        max_n_episodes=cfg.max_n_episodes,
        termination_penalty=0,
        seed=cfg.seed,
    )
    normalizer = dataset.normalizer

    flow_transformer = TransformerFlow(
        seq_len=cfg.horizon,
        in_channels=cfg.state_dim + cfg.action_dim,
        out_channels=cfg.state_dim + cfg.action_dim,
        hidden_size=cfg.transformer_config.hidden_size,
        depth=cfg.transformer_config.depth,
        num_heads=cfg.transformer_config.num_heads,
        mlp_ratio=cfg.transformer_config.mlp_ratio,
        x_emb_proj=cfg.transformer_config.x_emb_proj,
        x_emb_proj_conv_k=cfg.transformer_config.x_emb_proj_conv_k,
    ).to(cfg.device)
    flow_transformer.load_state_dict(torch.load(os.path.join(
        cfg.log_folder, cfg.env, 'flow', cfg.flow_exp_name, f'model_ema_{cfg.flow_cp}.pth'
    )))

    # get value model
    if cfg.guidance_method not in ['no']:
        value_model = ValueTransformer(
            input_dim=cfg.state_dim + cfg.action_dim,
            output_dim=1,
            model_dim=cfg.value_transformer_config.model_dim,
            num_heads=cfg.value_transformer_config.num_heads,
            num_layers=cfg.value_transformer_config.num_layers,
            dropout=cfg.value_transformer_config.dropout,
        ).to(cfg.device)
        value_model.load_state_dict(torch.load(os.path.join(
            cfg.log_folder, cfg.env, 'value', cfg.value_exp_name, f'model_{cfg.value_cp}.pth'
        )))
    else:
        value_model = None

    # get learned guidance model
    if cfg.guidance_method == 'guidance_matching':
        guide_model = TransformerFlow(
            seq_len=cfg.horizon,
            in_channels=cfg.state_dim + cfg.action_dim,
            out_channels=(cfg.state_dim + cfg.action_dim) if cfg.guide_matching_type != 'grad_z' else 1,
            hidden_size=cfg.guide_model_transformer_config.hidden_size,
            depth=cfg.guide_model_transformer_config.depth,
            num_heads=cfg.guide_model_transformer_config.num_heads,
            mlp_ratio=cfg.guide_model_transformer_config.mlp_ratio,
            x_emb_proj=cfg.guide_model_transformer_config.x_emb_proj,
            x_emb_proj_conv_k=cfg.guide_model_transformer_config.x_emb_proj_conv_k,
        ).to(cfg.device)
        if cfg.guide_matching_type != 'grad_z':
            guide_model.load_state_dict(torch.load(os.path.join(
                cfg.log_folder, cfg.env, 'guidance', cfg.guide_model_exp_name, f'model_{cfg.guide_matching_type}_{cfg.guide_model_cp}.pth'
            )))
        else:
            guide_model.load_state_dict(torch.load(os.path.join(
                cfg.log_folder, cfg.env, 'guidance', cfg.guide_model_exp_name, f'model_z_{cfg.guide_model_cp}.pth'
            )))
    else:
        guide_model = None

    flow_policy = FlowPolicy(
        flow_model=flow_transformer,
        value_model=value_model,
        guide_model=guide_model,
        normalizer=normalizer,
        action_dim=cfg.action_dim,
        state_dim=cfg.state_dim,
        horizon=cfg.horizon,
        cfg=cfg
    )

    # run env
    # TODO: make env parallel
    scores_ls = []
    for _ in range(cfg.random_repeat):
        score = run_env(dataset.env, flow_policy, cfg)
        scores_ls.append(score)
    
    # compute and save metrics
    scores = np.array(scores_ls)
    score = scores.mean()
    score_std = scores.std()
    
    with open(os.path.join(cfg.log_folder, cfg.env, 'eval', cfg.exp_name, 'results.txt'), 'a+') as f:
        f.write(f"score: {score} +- {score_std}\n")
        f.write(f"scores: {scores_ls}\n")

    # for vectorized env, compute mean and std total reward and score
    # save rollout and perhaps render



if __name__ == '__main__':
    cfg = tyro.cli(FlowMatchingEvaluationConfig)
    set_cuda_visible_device(cfg)
    deterministic(cfg.seed) # seed everything

    log_subfolder = os.path.join(cfg.log_folder, cfg.env, 'eval', cfg.exp_name)
    save_config(cfg, log_subfolder)

    evaluate(cfg)