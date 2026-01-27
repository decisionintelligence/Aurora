import argparse

from runner import AuroraRunner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-d', type=str,
                        help='Path to training data. (Folder contains data files, or data file)', default='/home/Aurora/text_image_general')
    parser.add_argument('--model_path', '-m', type=str, default='/home/Aurora/checkpoints/Aurora_Multi_Modal',
                        help='Path to pretrained model. Default: aurora/')
    parser.add_argument('--seq_len', type=int, default=528, help='Input Length')
    parser.add_argument('--pred_len', type=int, default=96, help='Output Length')

    parser.add_argument('--mode', type=str, default='fine_tune', choices=['pretrain', 'fine_tune'])

    parser.add_argument('--output_path', '-o', type=str, default='logs/aurora')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=0, help='minimum learning rate')

    parser.add_argument('--train_steps', type=int, default=None, help='number of training steps')
    parser.add_argument('--num_train_epochs', type=float, default=30, help='number of training epochs')

    parser.add_argument('--seed', type=int, default=5252, help='random seed')

    parser.add_argument('--lr_scheduler_type', type=str,
                        choices=['constant', 'linear', 'cosine', 'constant_with_warmup'], default='constant',
                        help='learning rate scheduler type')
    parser.add_argument('--warmup_ratio', type=float, default=0.3, help='warmup ratio')
    parser.add_argument('--warmup_steps', type=int, default=50000, help='warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')

    parser.add_argument('--global_batch_size', type=int, default=8192, help='global batch size')
    parser.add_argument('--micro_batch_size', type=int, default=1024, help='micro batch size per device')

    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], type=str, default='fp32',
                        help='precision mode (default: fp32)')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='enable gradient checkpointing')
    parser.add_argument('--deepspeed', type=str, default=None, help='DeepSpeed config file path')

    parser.add_argument('--save_steps', type=int, default=None, help='number of steps to save model')
    parser.add_argument('--save_strategy', choices=['steps', 'epoch', 'no'], type=str, default='epoch',
                        help='save strategy')
    parser.add_argument('--save_total_limit', type=int, default=None, help='limit the number of checkpoints')
    parser.add_argument('--save_only_model', action='store_true', help='save only model')

    parser.add_argument('--logging_steps', type=int, default=1, help='number of steps to log')
    parser.add_argument('--eval_strategy', choices=['steps', 'epoch', 'no'], type=str, default='no',
                        help='evaluation strategy')
    parser.add_argument('--eval_steps', type=int, default=None, help='number of evaluation steps')

    parser.add_argument('--adam_beta1', type=float, default=0.9, help='adam beta1')
    parser.add_argument('--adam_beta2', type=float, default=0.95, help='adam beta2')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='adam epsilon')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='max gradient norm')
    parser.add_argument('--dataloader_num_workers', type=int, default=10, help='number of workers for dataloader')

    args = parser.parse_args()

    runner = AuroraRunner(
        model_path=args.model_path,
        output_path=args.output_path,
        mode=args.mode,
        seed=args.seed,
    )

    runner.train_model(
        data_path=args.data_path,

        seq_len=args.seq_len,
        pred_len=args.pred_len,

        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,

        train_steps=args.train_steps,
        num_train_epochs=args.num_train_epochs,

        precision=args.precision,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        logging_steps=args.logging_steps,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.dataloader_num_workers,
        save_only_model=args.save_only_model,
        save_total_limit=args.save_total_limit,
    )
