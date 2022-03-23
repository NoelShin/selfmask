from argparse import ArgumentParser, Namespace
from math import ceil
import torch
from trainer import Trainer
from criterion import Loss
from evaluator import Evaluator
import clusterings
from utils.misc import get_dataset, get_lr_scheduler, get_model, set_seeds
from utils.visualizer import Visualizer
import wandb


def main(args: Namespace):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    torch.backends.cudnn.benchmark = True
    device: torch.device = torch.device("cuda:0")

    set_seeds(seed=args.seed)
    model = get_model(arch="maskformer", configs=args).to(device)
    model.train()

    dataset = get_dataset(
        dir_dataset=args.dir_dataset,
        dataset_name=args.dataset_name,
        mode="train",
        train_img_size=args.train_image_size,
        eval_img_size=args.eval_image_size,
        use_pseudo_masks=args.use_pseudo_masks,
        k=args.k,
        use_copy_paste=args.use_copy_paste,
        scale_range=args.scale_range,
        repeat_image=args.repeat_image,
        n_percent=args.n_percent if args.dataset_name == "imagenet1k" else None,
        n_copy_pastes=args.n_copy_pastes,
        pseudo_masks_fp=args.pseudo_masks_fp
    )

    model = get_model(arch="maskformer", configs=args).to(device)
    model.train()

    optimizer = torch.optim.AdamW([
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum
        }
    ])

    n_samples: int = len(dataset)
    n_iters_per_epoch = ceil(n_samples / args.batch_size)
    warmup_iters = int(args.n_epochs * n_iters_per_epoch * args.lr_warmup_duration / 100)

    lr_scheduler = get_lr_scheduler(
        optimizer=optimizer, n_epochs=args.n_epochs, n_iters_per_epoch=n_iters_per_epoch, warmup_iters=warmup_iters
    )
    print(
        f"\nLinear learning rate warmup for the first {warmup_iters} gradient iters "
        f"({args.lr_warmup_duration}% of total training iters).\n"
    )

    if args.clustering_mode == "spectral":
        clusterer = clusterings.SpectralClustering(use_gpu=args.use_gpu)
    else:
        clusterer = clusterings.KMeansClustering(use_gpu=args.use_gpu)

    criterion = Loss(
        weight_dice_loss=args.weight_dice_loss, weight_focal_loss=args.weight_focal_loss,
    )

    visualizer = Visualizer()

    evaluator = Evaluator(
        network=model,
        dir_dataset=args.dir_dataset,
        arch=args.arch,
        visualizer=visualizer,
        debug=args.debug,
    )

    trainer = Trainer(
        dataset=dataset,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        clusterer=clusterer,
        lr_scheduler=lr_scheduler,
        evaluator=evaluator,
        benchmarks=args.benchmarks,
        seed=args.seed,
        arch=args.arch,
        training_method=args.training_method,
        batch_size=args.batch_size,
        dir_ckpt=args.dir_ckpt,
        experim_name=args.experim_name,
        k=args.k,
        n_percent=args.n_percent if args.dataset_name == "imagenet1k" else None,
        scale_factor=args.scale_factor,
        eval_image_size=args.eval_image_size,
        visualizer=visualizer,
        debug=args.debug
    )
    trainer(args.n_epochs, device)


def define_experim_name(args: Namespace) -> str:
    list_keywords = list()
    list_keywords.append(f"nq{args.n_queries}_ndl{args.n_decoder_layers}")
    list_keywords.append("bc") if args.use_binary_classifier else None
    list_keywords.append("sup") if args.training_method == "supervised" else None
    list_keywords.append("p16") if args.patch_size == 16 and args.arch == "vit_small" else None
    list_keywords.append(f"sr{int(args.scale_range[0] * 100)}{int(args.scale_range[1] * 100)}")
    list_keywords.append(args.dataset_name)
    list_keywords.append("pm") if args.use_pseudo_masks else None
    list_keywords.append(f"seed{args.seed}")
    list_keywords.append(f"{args.suffix}") if args.suffix != '' else None
    return '_'.join(list_keywords)


if __name__ == '__main__':
    import os
    import json
    import yaml
    from argparse import Namespace

    parser = ArgumentParser("SelfMask")
    parser.add_argument("--config", type=str, default="", required=True)
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--overfitting", '-of', action="store_true", default=False)
    parser.add_argument("--seed", "-s", default=0, type=int)
    parser.add_argument("--suffix", type=str, default='')
    args: Namespace = parser.parse_args()
    base_args = yaml.safe_load(open(f"{args.config}", 'r'))

    args: dict = vars(args)
    args.update(base_args)
    args: Namespace = Namespace(**args)

    args.experim_name = define_experim_name(args)
    args.dir_ckpt = f"{args.dir_ckpt}/{args.experim_name}"
    os.makedirs(args.dir_ckpt, exist_ok=True)
    json.dump(vars(args), open(f"{args.dir_ckpt}/config.json", 'w'), indent=2, sort_keys=True)
    print(f"\n{args.dir_ckpt} is created.\n")

    # Weights & biases
    wandb.login()
    wandb.init(
        project=f"{args.experim_name}".replace(f"_seed{args.seed}", ''),
        name=f"seed_{args.seed}"
    )
    wandb.config.update(args)
    main(args)
    wandb.finish()
