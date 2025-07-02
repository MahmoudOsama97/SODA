# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from domainbed.lib import misc


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ['Debug28', 'RotatedMNIST', 'ColoredMNIST']

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.

    _hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet18', False, lambda r: False)
    _hparam('resnet50_augmix', True, lambda r: True)
    _hparam('dinov2', False, lambda r: False)
    _hparam('vit', False, lambda r: False)
    _hparam('vit_attn_tune', False, lambda r: False)
    _hparam('freeze_bn', False, lambda r: False)
    _hparam('lars', False, lambda r: False)
    _hparam('linear_steps', 500, lambda r: 500)
    _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    _hparam('vit_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    _hparam('class_balanced', False, lambda r: False)
    # TODO: nonlinear classifiers disabled
    _hparam('nonlinear_classifier', False,
            lambda r: bool(r.choice([False, False])))

    # Algorithm-specific hparam definitions. Each block of code below
    # corresponds to exactly one algorithm.

    if algorithm in ['DANN', 'CDANN']:
        _hparam('lambda', 1.0, lambda r: 10**r.uniform(-2, 2))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('d_steps_per_g_step', 1, lambda r: int(2**r.uniform(0, 3)))
        _hparam('grad_penalty', 0., lambda r: 10**r.uniform(-2, 1))
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('mlp_width', 256, lambda r: int(2 ** r.uniform(6, 10)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))

    elif algorithm == 'Fish':
        _hparam('meta_lr', 0.5, lambda r:r.choice([0.05, 0.1, 0.5]))

    elif algorithm == "RDM": 
        if dataset in ['DomainNet']: 
            _hparam('rdm_lambda', 0.5, lambda r: r.uniform(0.1, 1.0))
        elif dataset in ['PACS', 'TerraIncognita']:
            _hparam('rdm_lambda', 5.0, lambda r: r.uniform(1.0, 10.0))
        else:
            _hparam('rdm_lambda', 5.0, lambda r: r.uniform(0.1, 10.0))
            
        if dataset == 'DomainNet':
            _hparam('rdm_penalty_anneal_iters', 2400, lambda r: int(r.uniform(1500, 3000)))
        else:
            _hparam('rdm_penalty_anneal_iters', 1500, lambda r: int(r.uniform(800, 2700)))
            
        if dataset in ['TerraIncognita', 'OfficeHome', 'DomainNet']:
            _hparam('variance_weight', 0.0, lambda r: r.choice([0.0]))
        else:
            _hparam('variance_weight', 0.004, lambda r: r.uniform(0.001, 0.007))
            
        _hparam('rdm_lr', 1.5e-5, lambda r: r.uniform(8e-6, 2e-5))

    elif algorithm == "RSC":
        _hparam('rsc_f_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))
        _hparam('rsc_b_drop_factor', 1/3, lambda r: r.uniform(0, 0.5))

    elif algorithm == "SagNet":
        _hparam('sag_w_adv', 0.1, lambda r: 10**r.uniform(-2, 1))

    elif algorithm == "IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "Mixup":
        _hparam('mixup_alpha', 0.2, lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "GroupDRO":
        _hparam('groupdro_eta', 1e-2, lambda r: 10**r.uniform(-3, -1))

    elif algorithm == "MMD" or algorithm == "CORAL" or algorithm == "CausIRL_CORAL" or algorithm == "CausIRL_MMD":
        _hparam('mmd_gamma', 1., lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "MLDG":
        _hparam('mldg_beta', 1., lambda r: 10**r.uniform(-1, 1))
        _hparam('n_meta_test', 2, lambda r:  r.choice([1, 2]))

    elif algorithm == "MTL":
        _hparam('mtl_ema', .99, lambda r: r.choice([0.5, 0.9, 0.99, 1.]))

    elif algorithm == "VREx":
        _hparam('vrex_lambda', 1e1, lambda r: 10**r.uniform(-1, 5))
        _hparam('vrex_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "SD":
        _hparam('sd_reg', 0.1, lambda r: 10**r.uniform(-5, -1))

    elif algorithm == "ANDMask":
        _hparam('tau', 1, lambda r: r.uniform(0.5, 1.))

    elif algorithm == "IGA":
        _hparam('penalty', 1000, lambda r: 10**r.uniform(1, 5))

    elif algorithm == "SANDMask":
        _hparam('tau', 1.0, lambda r: r.uniform(0.0, 1.))
        _hparam('k', 1e+1, lambda r: 10**r.uniform(-3, 5))

    elif algorithm == "Fishr":
        _hparam('lambda', 1000., lambda r: 10**r.uniform(1., 4.))
        _hparam('penalty_anneal_iters', 1500, lambda r: int(r.uniform(0., 5000.)))
        _hparam('ema', 0.95, lambda r: r.uniform(0.90, 0.99))

    elif algorithm == "TRM":
        _hparam('cos_lambda', 1e-4, lambda r: 10 ** r.uniform(-5, 0))
        _hparam('iters', 200, lambda r: int(10 ** r.uniform(0, 4)))
        _hparam('groupdro_eta', 1e-2, lambda r: 10 ** r.uniform(-3, -1))

    elif algorithm == "IB_ERM":
        _hparam('ib_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('ib_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))
    elif algorithm == "OT_MarginalDomainAlign": # Or your chosen class name
        _hparam("lambda_ce", 1.0, lambda r: r.uniform(0.5, 1.5))
        _hparam("lambda_ot_align", 0.01, lambda r: r.choice([1e-4, 1e-3, 0.001, 0.1])) # Start much lower
        
        # OT specific parameters
        _hparam("ot_cost_metric", "sqeuclidean", lambda r: "sqeuclidean") # Usually fixed, not tuned
        _hparam("ot_epsilon", 0.1, lambda r: r.choice([0.01, 0.1, 0.5])) # Try a range 
        _hparam("ot_target_type", "pooled", lambda r: r.choice(['pooled', 'pairwise'])) # How to align
        _hparam("ot_p", 2, lambda r: 2) # Usually fixed to 2 for sqeuclidean, 1 for euclidean
        _hparam("ot_debias", True, lambda r: bool(r.choice([True, False])))    # Use debiased Sinkhorn divergence?

    elif algorithm == "OT_GromovWasserstein_Align": # Or your chosen class name
        _hparam("lambda_ce", 1.0, lambda r: r.uniform(0.5, 1.5))
        _hparam("lambda_ot_align", 0.01, lambda r: r.choice([1e-4, 1e-3, 0.001, 0.1])) # Start much lower
        # OT specific parameters
        _hparam("ot_target_type", "pooled", lambda r: r.choice(['pooled', 'pairwise'])) # How to align

    elif algorithm == "OT_AnnealedAlign": # Or your chosen class name
        _hparam("lambda_ce", 1.0, lambda r: r.uniform(0.5, 1.5))
        _hparam("lambda_ot_align", 0.01, lambda r: r.choice([1e-4, 1e-3, 0.001, 0.1])) # Start much lower
        
        # OT specific parameters
        _hparam("ot_cost_metric", "sqeuclidean", lambda r: "sqeuclidean") # Usually fixed, not tuned
        _hparam("ot_epsilon", 0.1, lambda r: r.choice([0.01, 0.1, 0.5])) # Try a range 
        _hparam("ot_target_type", "pooled", lambda r: r.choice(['pooled', 'pairwise'])) # How to align
        _hparam("ot_p", 2, lambda r: 2) # Usually fixed to 2 for sqeuclidean, 1 for euclidean
        _hparam("ot_debias", True, lambda r: bool(r.choice([True, False])))    # Use debiased Sinkhorn divergence?

    elif algorithm == "OT_PivotAlign": # Or your chosen class name
        _hparam("lambda_ce", 1.0, lambda r: r.uniform(0.5, 1.5))
        _hparam("lambda_ot_align", 0.01, lambda r: r.choice([1e-4, 1e-3, 0.001, 0.1])) # Start much lower
        
        # OT specific parameters
        _hparam("ot_cost_metric", "sqeuclidean", lambda r: "sqeuclidean") # Usually fixed, not tuned
        _hparam("ot_epsilon", 0.1, lambda r: r.choice([0.01, 0.1, 0.5])) # Try a range 
        _hparam("ot_target_type", "pooled", lambda r: r.choice(['pooled', 'pairwise'])) # How to align
        _hparam("ot_p", 2, lambda r: 2) # Usually fixed to 2 for sqeuclidean, 1 for euclidean
        _hparam("ot_debias", True, lambda r: bool(r.choice([True, False])))    # Use debiased Sinkhorn divergence?


    elif algorithm == "OT_EMAPivotAlign":
        # --- Core EMA Pivot Hyperparameters ---
        # Momentum for the EMA update of the pivot. Higher values mean a more stable target.
        _hparam("ema_momentum", 0.999, lambda r: r.choice([0.99, 0.999]))

        # Size (number of feature vectors) of the synthetic pivot domain.
        # This depends on available GPU memory.
        _hparam("ema_pivot_size", 128, lambda r: int(r.choice([128, 256])))

        # --- Optimal Transport Hyperparameters ---

        # Regularization strength for the Sinkhorn algorithm (passed as 'reg' to ot.sinkhorn2).
        # Controls the smoothness of the OT plan.
        _hparam("ot_reg", 0.1, lambda r: r.choice([0.05, 0.1, 0.5]))
        
        # --- Standard Loss Weight ---
        
        # Weight for the standard classification loss (usually fixed at 1.0).
        _hparam("lambda_ce", 1.0, lambda r: r.uniform(0.5, 1.5))
        _hparam("lambda_ot_align", 0.01, lambda r: r.choice([1e-4, 1e-3, 0.001, 0.1])) # Start much lower

    elif algorithm == "OT_EMAPivot_Confident":
        # --- Core EMA Pivot Hyperparameters ---

        # Momentum for the EMA update. Higher values mean a more stable target.
        _hparam("ema_momentum", 0.999, lambda r: r.choice([0.99, 0.999, 0.9999]))

        # Size (number of feature vectors) of the pivot queue.
        # This is usually a fixed architectural choice, not searched.
        _hparam("ema_pivot_size", 256, lambda r: 256)

        # --- NEW: Confidence threshold for updating the EMA queue ---
        # Only features from predictions with confidence > this value will update the pivot.
        _hparam("ema_confidence_gate", 0.9, lambda r: r.uniform(0.7, 0.95))

        # --- Optimal Transport Hyperparameters ---

        # Regularization strength for the Sinkhorn algorithm.
        # Note: The parameter name is 'ot_epsilon' in the algorithm code.
        _hparam("ot_epsilon", 0.1, lambda r: r.choice([0.01, 0.1, 0.5])) # Try a range 

        # --- Loss Weight Hyperparameters ---
        
        # Weight for the standard classification loss (usually fixed).
        _hparam("lambda_ce", 1.0, lambda r: r.uniform(0.5, 1.5))

        # Weight for the Optimal Transport alignment loss.
        _hparam("lambda_ot_align", 0.01, lambda r: r.choice([1e-4, 1e-3, 0.001, 0.1])) # Start much lower


    elif algorithm == "OT_EMAPivot_Confident_SSL":
        # --- Core EMA Pivot Hyperparameters ---

        # Momentum for the EMA update. Higher values mean a more stable target.
        _hparam("ema_momentum", 0.999, lambda r: r.choice([0.99, 0.999, 0.9999]))

        # Size (number of feature vectors) of the pivot queue.
        _hparam("ema_pivot_size", 256, lambda r: 256) # Usually fixed

        # --- NEW: Confidence threshold for updating the EMA queue ---
        # Only features from predictions with confidence > this value will update the pivot.
        _hparam("ema_confidence_gate", 0.9, lambda r: r.uniform(0.7, 0.95))

        # --- Optimal Transport Hyperparameters ---

        # Regularization strength for the Sinkhorn algorithm.
        _hparam("ot_epsilon", 0.1, lambda r: 10**r.uniform(-2, -0.3)) # Search from 0.01 to 0.5

        # --- Loss Weight Hyperparameters ---
        
        # Weight for the standard classification loss.
        _hparam("lambda_ce", 1.0, lambda r: r.uniform(0.5, 1.5))

        # Weight for the Optimal Transport alignment loss.
        _hparam("lambda_ot_align", 0.01, lambda r: r.choice([1e-4, 1e-3, 0.001, 0.1])) # Start much lower
        
        # --- NEW: Weight for the Self-Supervised (SSL) rotation loss ---
        _hparam("lambda_ssl", 0.5, lambda r: r.uniform(0.1, 1.0))




    elif algorithm == "OT_BarycentricPivotAlign":
        # --- Core Algorithm Hyperparameters ---
        # Weight for aligning to the stable, global EMA pivot.
        _hparam("lambda_ot_global", 0.1, lambda r: r.choice([0.0001, 0.001, 0.01, 0.1, 1]))
        # Weight for aligning to other domains within the minibatch.
        _hparam("lambda_ot_local", 0.1, lambda r: r.choice([0.0001, 0.001, 0.01, 0.1, 1]))
        # Momentum for the smooth pivot update. Higher is more stable.
        _hparam("ema_momentum", 0.999, lambda r: r.choice([0.99, 0.999]))
        # # Size of the pivot buffer.
        _hparam("pivot_size", 128, lambda r: int(r.choice([128, 256])))
        # Base regularization for Sinkhorn. This will be adapted by model confidence.
        _hparam("base_ot_reg", 0.1, lambda r: r.choice([0.05, 0.1, 0.5]))
        
        # --- Standard Loss Weight ---
        _hparam("lambda_ce", 1.0, lambda r: r.uniform(0.5, 1.5))

    elif algorithm == "OT_ProjectedPivotAlign":
        # Weight for the simplified global alignment loss.
        _hparam("lambda_ot_global", 0.1, lambda r: r.choice([0.0001, 0.001, 0.01, 0.1, 1]))
        # Momentum for the smooth pivot update.
        _hparam("ema_momentum", 0.999, lambda r: r.choice([0.99, 0.999]))
        # Size of the low-dimensional pivot buffer.
        _hparam("pivot_size", 128, lambda r: int(r.choice([128, 256])))
        # Regularization for Sinkhorn.
        _hparam("ot_reg", 0.1, lambda r: r.choice([0.05, 0.1, 0.5]))
        # Dimension of the projection head output.
        _hparam("projection_dim", 128, lambda r: int(r.choice([64, 128, 256])))
        # Standard classification loss weight.
        _hparam("lambda_ce", 1.0, lambda r: 1.0)

    elif algorithm == "OT_DualPivotAlign":
        # Weight for the dual OT alignment loss.
        _hparam("lambda_ot_global", 0.1, lambda r: r.choice([0.0001, 0.001, 0.01, 0.1, 1]))
        # Momentum for the smooth pivot update.
        _hparam("ema_momentum", 0.999, lambda r: r.choice([0.99, 0.999]))
        # Size of the pivot buffer.
        _hparam("pivot_size", 128, lambda r: int(r.choice([128, 256])))
        # Regularization for Sinkhorn.
        _hparam("ot_reg", 0.1, lambda r: r.choice([0.05, 0.1, 0.5, 1.0]))
        # Standard classification loss weight.
        _hparam("lambda_ce", 1.0, lambda r: 1.0)

    elif algorithm == "OT_FastPivotAlign":
        # Weight for the dual OT alignment loss.
        _hparam("lambda_ot_global", 0.1, lambda r: r.choice([0.0001, 0.001, 0.01, 0.1, 1]))
        # Momentum for the smooth pivot update.
        _hparam("ema_momentum", 0.999, lambda r: r.choice([0.99, 0.999]))
        # Size of the low-dimensional pivot buffer.
        _hparam("pivot_size", 128, lambda r: int(r.choice([128, 256])))
        # Regularization for Sinkhorn.
        _hparam("ot_reg", 0.1, lambda r: r.choice([0.1, 0.5, 1.0])) # Can be larger for low-dim
        # Dimension of the projection head output. THIS IS THE KEY SPEED KNOB.
        _hparam("projection_dim", 128, lambda r: int(r.choice([64, 128])))
        # Standard classification loss weight.
        _hparam("lambda_ce", 1.0, lambda r: 1.0)

    elif algorithm == "OT_GromovPivotAlign":
        # Weight for the GW alignment loss.
        _hparam("lambda_ot_global", 0.1, lambda r: r.choice([0.0001, 0.001, 0.01, 0.1, 1]))
        
        # Epsilon regularization for the entropic Gromov-Wasserstein solver.
        _hparam("ot_epsilon", 0.1, lambda r: r.choice([1, 5]))
        
        # The alignment strategy. 'pivot' is the main one for this idea.
        _hparam("ot_target_type", "pivot", lambda r: r.choice(['pivot', 'pooled', 'pairwise']))
        
        # Standard classification loss weight.
        _hparam("lambda_ce", 1.0, lambda r: r.uniform(0.5, 1.5))




    elif algorithm == "IB_IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))
        _hparam('ib_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('ib_penalty_anneal_iters', 500,
                lambda r: int(10**r.uniform(0, 4)))

    elif algorithm == "CAD" or algorithm == "CondCAD":
        _hparam('lmbda', 1e-1, lambda r: r.choice([1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]))
        _hparam('temperature', 0.1, lambda r: r.choice([0.05, 0.1]))
        _hparam('is_normalized', False, lambda r: False)
        _hparam('is_project', False, lambda r: False)
        _hparam('is_flipped', True, lambda r: True)

    elif algorithm == "Transfer":
        _hparam('t_lambda', 1.0, lambda r: 10**r.uniform(-2, 1))
        _hparam('delta', 2.0, lambda r: r.uniform(0.1, 3.0))
        _hparam('d_steps_per_g', 10, lambda r: int(r.choice([1, 2, 5])))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('gda', False, lambda r: True)
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))

    elif algorithm == 'EQRM':
        _hparam('eqrm_quantile', 0.75, lambda r: r.uniform(0.5, 0.99))
        _hparam('eqrm_burnin_iters', 2500, lambda r: 10 ** r.uniform(2.5, 3.5))
        _hparam('eqrm_lr', 1e-6, lambda r: 10 ** r.uniform(-7, -5))

    elif algorithm == 'ERMPlusPlus':
        _hparam('linear_lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    elif algorithm == 'URM':
        _hparam('urm', 'adversarial', lambda r: str(r.choice(['adversarial']))) # 'adversarial'
        
        _hparam('urm_adv_lambda', 0.1, lambda r: float(r.uniform(0,0.2)))
        _hparam('urm_discriminator_label_smoothing', 0, lambda r: float(r.uniform(0, 0)))
        _hparam('urm_discriminator_optimizer', 'adam', lambda r: str(r.choice(['adam'])))
        _hparam('urm_discriminator_hidden_layers', 1, lambda r: int(r.choice([1,2,3])))
        _hparam('urm_generator_output', 'tanh', lambda r: str(r.choice(['tanh', 'relu'])))
                
        if dataset in SMALL_IMAGES:
            _hparam('urm_discriminator_lr', 1e-3, lambda r: 10**r.uniform(-5.5, -3.5))
        else:
            _hparam('urm_discriminator_lr', 5e-5, lambda r: 10**r.uniform(-6, -4.5))


    if algorithm == "ADRMX":
        _hparam('cnt_lambda', 1.0, lambda r: r.choice([1.0]))
        _hparam('dclf_lambda', 1.0, lambda r: r.choice([1.0]))
        _hparam('disc_lambda', 0.75, lambda r: r.choice([0.75]))
        _hparam('rmxd_lambda', 1.0, lambda r: r.choice([1.0]))
        _hparam('d_steps_per_g_step', 2, lambda r: r.choice([2]))
        _hparam('beta1', 0.5, lambda r: r.choice([0.5]))
        _hparam('mlp_width', 256, lambda r: r.choice([256]))
        _hparam('mlp_depth', 9, lambda r: int(r.choice([8, 9, 10])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0]))


    # Dataset-and-algorithm-specific hparam definitions. Each block of code
    # below corresponds to exactly one hparam. Avoid nested conditionals.

    if dataset in SMALL_IMAGES:
        if algorithm == "ADRMX":
            _hparam('lr', 3e-3, lambda r: r.choice([5e-4, 1e-3, 2e-3, 3e-3]))
        else:
            _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    else:
        if algorithm == "ADRMX":
            _hparam('lr', 3e-5, lambda r: r.choice([2e-5, 3e-5, 4e-5, 5e-5]))
        else:
            _hparam('lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if dataset in SMALL_IMAGES:
        _hparam('weight_decay', 0., lambda r: 0.)
    else:
        _hparam('weight_decay', 0., lambda r: 10**r.uniform(-6, -2))

    if dataset in SMALL_IMAGES:
        _hparam('batch_size', 64, lambda r: int(2**r.uniform(3, 9)))
    elif algorithm == 'ARM':
        _hparam('batch_size', 8, lambda r: 8)
    elif algorithm == 'RDM':
        if dataset in ['DomainNet', 'TerraIncognita']:
            _hparam('batch_size', 40, lambda r: int(r.uniform(30, 60)))
        else:
            _hparam('batch_size', 88, lambda r: int(r.uniform(70, 100)))
    elif dataset == 'DomainNet':
        _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5)))
    else:
        _hparam("batch_size", 32, lambda r: 32)

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_g', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_g', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('lr_d', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if algorithm in ['DANN', 'CDANN'] and dataset in SMALL_IMAGES:
        _hparam('weight_decay_g', 0., lambda r: 0.)
    elif algorithm in ['DANN', 'CDANN']:
        _hparam('weight_decay_g', 0., lambda r: 10**r.uniform(-6, -2))

    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
