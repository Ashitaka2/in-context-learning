import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model
import numpy as np

import wandb

torch.backends.cudnn.benchmark = True

def strip_module_prefix(state_dict):
    from collections import OrderedDict
    return OrderedDict(
        (k.replace("module.", "") if k.startswith("module.") else k, v)
        for k, v in state_dict.items()
    )


def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    # n_dims = model.n_dims
    # Modified n_dims access 
    n_dims = model.module.n_dims if isinstance(model, torch.nn.DataParallel) else model.n_dims


    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)

        loss_func = task.get_training_metric()

        loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func)

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            raw_state_dict = strip_module_prefix(model.state_dict())
            training_state = {
                # "model_state_dict": model.state_dict(),
                "model_state_dict": raw_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            # print(f"model.state.dict is {model.state_dict()}")
            raw_state_dict = strip_module_prefix(model.state_dict())
            torch.save(raw_state_dict, os.path.join(args.out_dir, f"model_{i}.pt"))
            # torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )
    
    args.gpu = list(range(args.gpu_start, args.gpu_start + args.gpu_num))
    # print(f"args.gpu is {gpu}")
    if isinstance(args.gpu, list):
        device = torch.device(f"cuda:{args.gpu[0]}")
        available_gpus = ','.join(map(str, args.gpu))
        torch.cuda.set_device(device)
        print(f"Using GPUs: {available_gpus}")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    model = build_model(args.model)
    if isinstance(args.gpu, list) and len(args.gpu) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
    print("device: ", device)
    model.to(device)
    # model.cuda()

    model.train()
    train(model, args)

    if not args.test_run:
        # _ = get_run_metrics(args.out_dir)  # precompute metrics for eval
        eval_metrics = get_run_metrics(args.out_dir) 


    # wandb metric record
    if args.wandb and not args.test_run:
        eval_metrics = eval_metrics['standard']
        eval_models = list(eval_metrics.keys())
        plot_y = []

        val_acc = eval_metrics[model.module.name if isinstance(model, torch.nn.DataParallel) else model.name]['mean']
        mean_val_acc = np.mean(val_acc)

        wandb.log({"mean_val_acc": mean_val_acc})

        for model_name in eval_models:
            plot_y.append(eval_metrics[model_name]['mean'])
        plot_x = list(range(len(plot_y[0])))

        wandb.log({
            'eval/mean_acc': wandb.plot.line_series(
                plot_x,
                plot_y,
                keys=eval_models,
                title='Accuracy of Different Models',
                xname='Incontext Examples'
            )
        })
    


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
