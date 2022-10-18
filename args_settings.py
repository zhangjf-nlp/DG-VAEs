# Definition of parameters for experiments
# for VAE and DG-VAE on Yahoo dataset and Yelp Dataset, you can run the two experiments in terminal:
# e.g.,
#    python3 experiment.py --dataset yahoo --encoder_class GaussianLSTMEncoder
#    python3 experiment.py --dataset yelp --encoder_class GaussianLSTMEncoder
#    python3 experiment.py --dataset yahoo --encoder_class DGGaussianLSTMEncoder
#    python3 experiment.py --dataset yelp --encoder_class DGGaussianLSTMEncoder
#
# however, we include more than 100 experiments, so automatic iteration of experiment arguments settings is needed
# e.g.,
#    args_specifications = experiments_args_specifications["strategy_or_structure"]
#    one_key_experiment(args_specifications) # defined in **one_key_experiment.ipynb**
#    experiments_evaluation_when_all_complete(args_specifications) # defined in **watch_experiments.ipynb**
#
# we also provide api for combination of multiple groups of settings
# e.g.,
#    args_specifications = select_args_specifications(["strategy_or_structure","restriction_or_weakened_kl"], only_small=True)
#    one_key_experiment(args_specifications) # defined in **one_key_experiment.ipynb**
#    experiments_evaluation_when_all_complete(args_specifications) # defined in **watch_experiments.ipynb**
# only_small=True means only include experiments on short datasets, i.e., Short-Yelp and SNLI

experiments_args_specifications = {}

experiments_args_specifications["restriction_or_weakened_kl"] = [
    [
        "--dataset", str(dataset),
        "--encoder_class", str(encoder_class)
    ] for dataset in [
        "short_yelp", "snli", "yahoo", "yelp"
    ] for encoder_class in [
        "GaussianLSTMEncoder", # VAE (default)
        "BNGaussianLSTMEncoder", # BN-VAE (0.7)
        "DeltaGaussianLSTMEncoder", # Delta-VAE (0.15)
        "FineFBGaussianLSTMEncoder", # FB-VAE (4)
        "DGGaussianLSTMEncoder", # proposed DG-VAE (default)
        "VMFLSTMEncoder", # vMF-VAE, \kappa=13
        "DGVMFLSTMEncoder", # proposed DG-vMF-VAE (default), \kappa=13
    ]
]

experiments_args_specifications["strategy_or_structure"] = [
    [
        "--dataset", str(dataset),
        "--encoder_class", str(encoder_class),
        str(additional_specification_1),
        str(additional_specification_2),
    ] for dataset in [
        "short_yelp", "snli", "yahoo", "yelp"
    ] for encoder_class in [
        "GaussianLSTMEncoder",
    ] for additional_specification_1, additional_specification_2 in [
        ("", ""), # onetime 10 epochs annealing, i.e., VAE (default)
        ("--cycle", 20), # cyclic 10 epochs annealing for every 20 epochs, i.e., Cyclic-VAE
        ("--add_skip", ""), # skip-vae
        ("--add_bow", ""), # bow-vae
    ]
]

# experiments on Beta-VAE with different values of kl_beta
experiments_args_specifications["kl_beta"] = [
    [
        "--dataset", str(dataset),
        "--encoder_class", str(encoder_class),
        "--kl_beta", str(kl_beta)
    ] for dataset in [
        "short_yelp", "snli", "yahoo", "yelp"
    ] for encoder_class in [
        "GaussianLSTMEncoder",
    ] for kl_beta in [
        0.0, 0.1, 0.2, 0.4, 0.8, 1.0
    ]
]


# experiments on BN-VAE with different values of gamma
experiments_args_specifications["bn_gamma"] = [
    [
        "--dataset", str(dataset),
        "--encoder_class", str(encoder_class),
        "--gamma", str(gamma)
    ] for dataset in [
        "short_yelp", "snli", "yahoo", "yelp"
    ] for encoder_class in [
        "BNGaussianLSTMEncoder",
    ] for gamma in [
        0.6, #gamma=0.60 -> KL>=5.76
        0.7, #gamma=0.70 -> KL>=7.84
        0.9, #gamma=0.90 -> KL>=12.96
        1.2, #gamma=1.20 -> KL>=23.04
        1.5, #gamma=1.50 -> KL>=36.00
        1.8, #gamma=1.80 -> KL>=51.84
    ]
]

# experiments on FB-VAE with different values of target_kl
experiments_args_specifications["fb_target_kl"] = [
    [
        "--dataset", str(dataset),
        "--encoder_class", str(encoder_class),
        "--target_kl", str(target_kl)
    ] for dataset in [
        "short_yelp", "snli", "yahoo", "yelp"
    ] for encoder_class in [
        "FineFBGaussianLSTMEncoder",
    ] for target_kl in [
        4, 9, 16, 25, 36, 49
    ]
]

# experiments on proposed DG-VAE with different values of agg_size, i.e., ablation study
experiments_args_specifications["agg_size"] = [
    [
        "--dataset", str(dataset),
        "--encoder_class", str(encoder_class),
        "--agg_size", str(agg_size)
    ] for dataset in [
        "short_yelp", "snli", "yahoo", "yelp"
    ] for encoder_class in [
        "DGGaussianLSTMEncoder",
    ] for agg_size in [
        1, 2, 4, 8, 16, 32
    ]
]

# experiments on vMF-VAE and DG-vMF-VAE with different values of kappa
experiments_args_specifications["vmf_kappa"] = [
    [
        "--dataset", str(dataset),
        "--encoder_class", str(encoder_class),
        "--kappa", str(kappa)
    ] for dataset in [
        "short_yelp", "snli", "yahoo", "yelp"
    ] for encoder_class in [
        "VMFLSTMEncoder",
        "DGVMFLSTMEncoder",
    ] for kappa in [
        13,  #kappa=13, kl=2.172591324079198
        25,  #kappa=25, kl=5.755404266812285
        50,  #kappa=50, kl=12.253369028271255
        100, #kappa=100, kl=20.75852807931199
        200, #kappa=200, kl=30.373301389208724
    ]
]

def select_args_specifications(names:list, only_small=False):
    args_specifications = [_ for name in names for _ in experiments_args_specifications[name]]
    if only_small:
        args_specifications = [_ for _ in args_specifications if not any([dataset in _ for dataset in ["yahoo", "yelp"]])]
    return args_specifications