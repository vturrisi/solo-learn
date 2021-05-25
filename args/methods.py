def barlow_twins_args(parser):
    # projector
    parser.add_argument("--encoding_dim", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=2048)

    # parameters
    parser.add_argument("--lamb", type=float, default=5e-3)
    parser.add_argument("--scale_loss", type=float, default=0.025)


def byol_args(parser):
    # projector
    parser.add_argument("--encoding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=2048)

    # predictor
    parser.add_argument("--pred_hidden_dim", type=int, default=512)

    # momentum settings
    parser.add_argument("--base_tau_momentum", default=0.99, type=float)
    parser.add_argument("--final_tau_momentum", default=1.0, type=float)


def mocov2plus_args(parser):
    # projector
    parser.add_argument("--encoding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=2048)

    # parameters
    parser.add_argument("--temperature", type=float, default=0.1)

    # queue settings
    parser.add_argument("--queue_size", default=65536, type=int)

    # momentum settings
    parser.add_argument("--base_tau_momentum", default=0.99, type=float)
    parser.add_argument("--final_tau_momentum", default=1.0, type=float)


def nnclr_args(parser):
    # projector
    parser.add_argument("--encoding_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=2048)

    # predictor
    parser.add_argument("--pred_hidden_dim", type=int, default=4096)

    # queue settings
    parser.add_argument("--queue_size", default=65536, type=int)

    # parameters
    parser.add_argument("--temperature", type=float, default=0.2)


def simclr_args(parser):
    # projector
    parser.add_argument("--encoding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=2048)

    # parameters
    parser.add_argument("--temperature", type=float, default=0.1)

    # supervised-simclr
    parser.add_argument("--supervised", action="store_true")


def simsiam_args(parser):
    # projector
    parser.add_argument("--encoding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=2048)

    # predictor
    parser.add_argument("--pred_hidden_dim", type=int, default=512)


def swav_args(parser):
    # projector
    parser.add_argument("--encoding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=2048)

    # queue settings
    parser.add_argument("--queue_size", default=3840, type=int)

    # parameters
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num_prototypes", type=int, default=3000)
    parser.add_argument("--sk_epsilon", type=float, default=0.05)
    parser.add_argument("--sk_iters", type=int, default=3)
    parser.add_argument("--freeze_prototypes_epochs", type=int, default=1)
    parser.add_argument("--epoch_queue_starts", type=int, default=15)


def vicreg_args(parser):
    # projector
    parser.add_argument("--encoding_dim", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=2048)

    # parameters
    parser.add_argument("--sim_loss_weight", default=25, type=float)
    parser.add_argument("--var_loss_weight", default=25, type=float)
    parser.add_argument("--cov_loss_weight", default=1.0, type=float)
