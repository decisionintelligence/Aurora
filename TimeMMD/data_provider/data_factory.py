from data_provider.data_loader import Dataset_TimeMMD
from torch.utils.data import DataLoader

data_dict = {
    'Agriculture': Dataset_TimeMMD,
    'Climate': Dataset_TimeMMD,
    'Economy': Dataset_TimeMMD,
    'Energy': Dataset_TimeMMD,
    'Environment': Dataset_TimeMMD,
    'Health': Dataset_TimeMMD,
    'Security': Dataset_TimeMMD,
    'Traffic': Dataset_TimeMMD,
    'SocialGood': Dataset_TimeMMD,
    'KR': Dataset_TimeMMD,
    'EWJ': Dataset_TimeMMD,
    'MDT': Dataset_TimeMMD,
    'Weather': Dataset_TimeMMD
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
