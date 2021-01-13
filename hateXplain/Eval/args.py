
def add_eval_args(parser):

    parser.add_argument('model_to_use',
                           metavar='--model_to_use',
                           type=str,
                           help='model to use for evaluation')

    parser.add_argument('attention_lambda',
                           metavar='--attention_lambda',
                           type=float,
                           help='required to assign the contribution of the atention loss')

    parser.add_argument('--num_supervised_heads',
                           type=int,
                           default=None,
                           help='Number of supervised heads (BERT variants only)')

    parser.add_argument('--subset',
                           type=str,
                           default='test',
                           help='Subset of data to evaluate on')

    return parser
